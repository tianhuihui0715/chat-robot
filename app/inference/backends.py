from __future__ import annotations

import asyncio
import json
import re
from typing import Protocol

from app.schemas.chat import ChatMessage, IntentDecision
from app.schemas.inference import (
    InferenceGenerateRequest,
    InferenceIntentRequest,
)


class InferenceBackend(Protocol):
    @property
    def model_name(self) -> str | None:
        ...

    @property
    def model_loaded(self) -> bool:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


class GenerationInferenceBackend(InferenceBackend, Protocol):
    async def generate(self, request: InferenceGenerateRequest) -> str:
        ...


class IntentInferenceBackend(InferenceBackend, Protocol):
    async def decide(self, request: InferenceIntentRequest) -> tuple[IntentDecision, str]:
        ...


class MockInferenceBackend:
    @property
    def model_name(self) -> str | None:
        return "mock-inference-backend"

    @property
    def model_loaded(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, request: InferenceGenerateRequest) -> str:
        latest_user_message = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        if request.sources:
            source_titles = ", ".join(source.title for source in request.sources[:3])
            return (
                "这是独立推理服务返回的 mock 回答。"
                f"问题：{latest_user_message}。参考片段：{source_titles}。"
            )
        return f"这是独立推理服务返回的 mock 回答。问题：{latest_user_message}"


class MockIntentInferenceBackend:
    @property
    def model_name(self) -> str | None:
        return "mock-intent-backend"

    @property
    def model_loaded(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def decide(self, request: InferenceIntentRequest) -> tuple[IntentDecision, str]:
        decision = _heuristic_intent_decision(request.messages)
        return decision, decision.model_dump_json(ensure_ascii=False)


class LocalHFGenerationBackend:
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int,
        default_max_new_tokens: int,
    ) -> None:
        self._model_path = model_path
        self._max_input_tokens = max_input_tokens
        self._default_max_new_tokens = default_max_new_tokens
        self._tokenizer = None
        self._model = None
        self._torch = None

    @property
    def model_name(self) -> str | None:
        return self._model_path

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def start(self) -> None:
        self._torch, self._tokenizer, self._model = _load_quantized_model(self._model_path)

    async def stop(self) -> None:
        _release_model(self._torch, self._tokenizer, self._model)
        self._tokenizer = None
        self._model = None
        self._torch = None

    async def generate(self, request: InferenceGenerateRequest) -> str:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("LocalHFGenerationBackend has not been started.")

        chat_messages = self._build_chat_messages(request)
        answer = _generate_chat_completion(
            tokenizer=self._tokenizer,
            model=self._model,
            torch_module=self._torch,
            chat_messages=chat_messages,
            max_input_tokens=self._max_input_tokens,
            max_new_tokens=request.max_new_tokens or self._default_max_new_tokens,
        )
        return _sanitize_output(answer)

    def _build_chat_messages(self, request: InferenceGenerateRequest) -> list[dict[str, str]]:
        system_parts = [
            "你是一个本地部署的中文 AI 助手。",
            "请直接给出答案，不要输出思考过程，也不要输出 <think> 标签。",
        ]

        if request.sources:
            source_sections = []
            for index, source in enumerate(request.sources, start=1):
                source_sections.append(
                    f"[{index}] id={source.document_id} title={source.title}\n{source.content}"
                )
            system_parts.append(
                "回答时优先参考以下知识片段；如果依据不足，请明确说明。\n\n"
                + "\n\n".join(source_sections)
            )

        extra_system_messages = [
            message.content.strip()
            for message in request.messages
            if message.role == "system" and message.content.strip()
        ]
        if extra_system_messages:
            system_parts.append("额外约束：\n" + "\n".join(extra_system_messages))

        chat_messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ]
        chat_messages.extend(
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
            if message.role in {"user", "assistant"}
        )
        return chat_messages


class LocalHFIntentBackend:
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int,
        prompt_role: str,
        prompt_task: str,
        available_intents: list[str],
        decision_rules: list[str],
        rewrite_rules: list[str],
        rationale_rule: str,
        output_schema: str,
        examples: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> None:
        self._model_path = model_path
        self._max_input_tokens = max_input_tokens
        self._max_new_tokens = max_new_tokens
        self._prompt_role = prompt_role
        self._prompt_task = prompt_task
        self._available_intents = available_intents
        self._decision_rules = decision_rules
        self._rewrite_rules = rewrite_rules
        self._rationale_rule = rationale_rule
        self._output_schema = output_schema
        self._examples = examples
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str | None:
        return self._model_path

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        async with self._load_lock:
            self._unload()

    async def decide(self, request: InferenceIntentRequest) -> tuple[IntentDecision, str]:
        rule_decision = _rule_based_intent_decision(request.messages)
        if rule_decision is not None:
            return rule_decision, rule_decision.model_dump_json(ensure_ascii=False)

        latest_user_message = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        ).strip()
        if _should_route_to_task_or_qa(latest_user_message):
            binary_decision, raw_output = await self._decide_task_or_qa(request.messages)
            if binary_decision is not None:
                return binary_decision, raw_output

        async with self._load_lock:
            await self._ensure_loaded()

            if self._model is None or self._tokenizer is None or self._torch is None:
                raise RuntimeError("LocalHFIntentBackend failed to load the intent model.")

            raw_output = _generate_chat_completion(
                tokenizer=self._tokenizer,
                model=self._model,
                torch_module=self._torch,
                chat_messages=self._build_chat_messages(request.messages),
                max_input_tokens=self._max_input_tokens,
                max_new_tokens=self._max_new_tokens,
            )

        cleaned_output = _sanitize_output(raw_output)
        decision = _parse_intent_decision(cleaned_output, request.messages)
        return decision, cleaned_output

    async def _decide_task_or_qa(
        self,
        messages: list[ChatMessage],
    ) -> tuple[IntentDecision | None, str]:
        async with self._load_lock:
            await self._ensure_loaded()

            if self._model is None or self._tokenizer is None or self._torch is None:
                raise RuntimeError("LocalHFIntentBackend failed to load the intent model.")

            raw_output = _generate_chat_completion(
                tokenizer=self._tokenizer,
                model=self._model,
                torch_module=self._torch,
                chat_messages=_build_task_or_qa_messages(messages),
                max_input_tokens=self._max_input_tokens,
                max_new_tokens=16,
            )

        cleaned_output = _sanitize_output(raw_output)
        label = _parse_task_or_qa_label(cleaned_output)
        if label is None:
            return None, cleaned_output

        latest_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        ).strip()
        if label == "task":
            decision = IntentDecision(
                intent="task",
                need_rag=False,
                rewrite_query=latest_user_message,
                rationale="规则粗筛后由模型二分类判定为 task。",
            )
        else:
            decision = IntentDecision(
                intent="knowledge_qa",
                need_rag=True,
                rewrite_query=_normalize_knowledge_query(latest_user_message),
                rationale="规则粗筛后由模型二分类判定为 knowledge_qa。",
            )
        return decision, cleaned_output

    async def _ensure_loaded(self) -> None:
        if self.model_loaded:
            return
        self._torch, self._tokenizer, self._model = _load_cpu_model(self._model_path)

    def _unload(self) -> None:
        _release_model(self._torch, self._tokenizer, self._model)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        chat_messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
        ]

        examples_prompt = self._build_examples_prompt()
        if examples_prompt:
            chat_messages.append({"role": "system", "content": examples_prompt})

        chat_messages.append(
            {
                "role": "user",
                "content": "请对下面的对话做意图识别，只返回 JSON。\n"
                + _format_conversation(messages),
            }
        )
        return chat_messages

    def _build_system_prompt(self) -> str:
        lines = [
            self._prompt_role.strip(),
            f"任务：{self._prompt_task.strip()}",
            "可选 intent 只有：" + ", ".join(self._available_intents) + "。",
            "判断规则：",
        ]
        lines.extend(f"- {rule.strip()}" for rule in self._decision_rules if rule.strip())
        lines.append("rewrite_query 规则：")
        lines.extend(f"- {rule.strip()}" for rule in self._rewrite_rules if rule.strip())
        lines.append(self._rationale_rule.strip())
        lines.append("严格只输出 JSON，格式如下：")
        lines.append(self._output_schema.strip())
        return "\n".join(lines)

    def _build_examples_prompt(self) -> str:
        sections: list[str] = []
        for index, example in enumerate(self._examples, start=1):
            conversation = example.get("conversation", "").strip()
            output = example.get("output", "").strip()
            if not conversation or not output:
                continue
            sections.append(f"示例{index}:\n输入对话:\n{conversation}\n输出:\n{output}")
        return "\n\n".join(sections)


def _load_quantized_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return torch, tokenizer, model


def _load_cpu_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return torch, tokenizer, model


def _release_model(torch_module, tokenizer, model) -> None:
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _generate_chat_completion(
    tokenizer,
    model,
    torch_module,
    chat_messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        attention_mask = attention_mask[:, -max_input_tokens:]

    device = model.device
    with torch_module.inference_mode():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][input_ids.shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _sanitize_output(answer: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip() or answer.strip()


def _format_conversation(messages: list[ChatMessage]) -> str:
    return json.dumps(
        [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        ensure_ascii=False,
    )


def _heuristic_intent_decision(messages: list[ChatMessage]) -> IntentDecision:
    rag_keywords = (
        "文档",
        "知识库",
        "部署",
        "说明",
        "配置",
        "接口",
        "怎么",
        "如何",
        "查询",
        "项目",
    )
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    need_rag = any(keyword in latest_user_message for keyword in rag_keywords)

    if need_rag:
        intent = "knowledge_qa"
        rationale = "命中了知识问答关键词，优先尝试走检索增强生成。"
    elif len(messages) > 1:
        intent = "follow_up"
        rationale = "检测到多轮上下文，先按追问处理。"
    else:
        intent = "chat"
        rationale = "未命中检索关键词，按普通对话处理。"

    return IntentDecision(
        intent=intent,
        need_rag=need_rag,
        rewrite_query=latest_user_message,
        rationale=rationale,
    )


def _parse_intent_decision(output: str, messages: list[ChatMessage]) -> IntentDecision:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    fallback = _heuristic_intent_decision(messages)

    try:
        data = json.loads(_extract_json_object(output))
        decision = IntentDecision.model_validate(data)
    except Exception:
        return fallback

    rewrite_query = _normalize_rewrite_query(
        rewrite_query=decision.rewrite_query,
        latest_user_message=latest_user_message,
        messages=messages,
        intent=decision.intent,
    )
    rationale = decision.rationale.strip() or fallback.rationale
    need_rag = decision.need_rag if decision.intent != "reject" else False

    return IntentDecision(
        intent=decision.intent,
        need_rag=need_rag,
        rewrite_query=rewrite_query,
        rationale=rationale,
    )


def _normalize_rewrite_query(
    rewrite_query: str,
    latest_user_message: str,
    messages: list[ChatMessage],
    intent: str,
) -> str:
    candidate = rewrite_query.strip()
    if not candidate:
        candidate = latest_user_message
    if _looks_like_generic_question(candidate):
        candidate = latest_user_message
    if len(candidate) <= 2:
        candidate = latest_user_message
    if not _has_meaningful_overlap(candidate, latest_user_message):
        candidate = latest_user_message
    if intent == "follow_up":
        candidate = _expand_follow_up_query(candidate, messages)
    return candidate


def _looks_like_generic_question(text: str) -> bool:
    normalized = text.replace(" ", "")
    generic_phrases = (
        "您有什么问题吗",
        "你有什么问题吗",
        "请提供更多信息",
        "请说明您的问题",
        "请描述您的需求",
        "有什么可以帮您",
    )
    return any(phrase in normalized for phrase in generic_phrases)


def _has_meaningful_overlap(candidate: str, original: str) -> bool:
    candidate_tokens = _extract_keywords(candidate)
    original_tokens = _extract_keywords(original)
    if not original_tokens:
        return True
    return bool(candidate_tokens & original_tokens)


def _extract_keywords(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]{2,}", text))
    return {token.lower() for token in tokens if token.strip()}


def _expand_follow_up_query(candidate: str, messages: list[ChatMessage]) -> str:
    if not _looks_like_follow_up_fragment(candidate):
        return candidate

    previous_user = ""
    for message in reversed(messages[:-1]):
        if message.role == "user" and message.content.strip():
            previous_user = message.content.strip()
            break

    if not previous_user:
        return candidate

    return f"{previous_user}；补充问题：{candidate}"


def _looks_like_follow_up_fragment(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) <= 10:
        return True

    fragment_markers = (
        "那",
        "这个",
        "这个呢",
        "那个",
        "那这个",
        "那这个呢",
        "那 Windows",
        "详细说说",
        "继续",
        "为什么",
        "怎么配",
        "怎么做",
        "放哪",
    )
    return any(normalized.startswith(marker) for marker in fragment_markers)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def _rule_based_intent_decision(messages: list[ChatMessage]) -> IntentDecision | None:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not latest_user_message:
        return None

    if _matches_reject_rule(latest_user_message):
        return IntentDecision(
            intent="reject",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中高风险请求规则，判定为 reject。",
        )

    if _matches_chat_rule(latest_user_message):
        return IntentDecision(
            intent="chat",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中问候或闲聊规则，判定为 chat。",
        )

    if _looks_like_follow_up_by_rules(messages):
        expanded_query = _expand_follow_up_query(latest_user_message, messages)
        need_rag = _follow_up_needs_rag(expanded_query, messages)
        return IntentDecision(
            intent="follow_up",
            need_rag=need_rag,
            rewrite_query=expanded_query,
            rationale="命中上下文追问规则，判定为 follow_up。",
        )

    if _matches_task_rule(latest_user_message):
        return IntentDecision(
            intent="task",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中任务执行规则，判定为 task。",
        )

    if _matches_knowledge_rule(latest_user_message):
        return IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query=_normalize_knowledge_query(latest_user_message),
            rationale="命中知识问答规则，判定为 knowledge_qa。",
        )

    return None


def _matches_reject_rule(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if not normalized:
        return False

    reject_patterns = (
        r"制作.*炸弹",
        r"炸弹.*制作",
        r"窃取.*(账号|账[号戶]|密码|凭证)",
        r"(盗取|窃取).*(密码|账号|隐私|凭证)",
        r"入侵.*(服务器|系统|网站)",
        r"骗取.*(隐私|信息|资料)",
        r"(木马|勒索|钓鱼).*(脚本|程序|代码)",
    )
    reject_keywords = (
        "炸弹",
        "窃取账号",
        "窃取密码",
        "入侵服务器",
        "骗取隐私",
        "恶意脚本",
        "钓鱼话术",
    )
    return any(keyword in normalized for keyword in reject_keywords) or any(
        re.search(pattern, normalized) for pattern in reject_patterns
    )


def _matches_chat_rule(text: str) -> bool:
    normalized = text.strip().lower()
    if len(normalized) > 30:
        return False

    chat_patterns = (
        r"^(你好|您好|嗨|hi|hello)[！!。,.，]*$",
        r"^(谢谢|感谢|辛苦了|多谢)[！!。,.，]*$",
        r"^(再见|拜拜|晚安)[！!。,.，]*$",
        r"^你(是谁|能做什么).*$",
        r"^你.*(能帮我做什么|可以帮我做什么|都能做什么).*$",
    )
    return any(re.match(pattern, normalized) for pattern in chat_patterns)


def _should_route_to_task_or_qa(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False

    if _matches_chat_rule(text) or _matches_task_rule(text) or _matches_knowledge_rule(text):
        return False

    task_keywords = (
        "帮我",
        "请帮我",
        "生成",
        "写一个",
        "写一份",
        "写个",
        "改写",
        "整理",
        "总结",
        "翻译",
        "列出",
        "制定",
        "设计",
        "规划",
        "提纲",
        "模板",
    )
    qa_keywords = (
        "什么",
        "怎么",
        "为什么",
        "哪里",
        "多少",
        "区别",
        "作用",
        "方式",
        "配置",
        "接口",
        "部署",
        "文档",
        "说明",
        "参数",
    )
    return any(keyword in normalized for keyword in task_keywords) or any(
        keyword in normalized for keyword in qa_keywords
    )


def _matches_task_rule(text: str) -> bool:
    normalized = text.strip()
    task_patterns = (
        r"^帮我(写|生成|整理|总结|改写|翻译|列出|制定|设计|规划).+",
        r"^请(帮我)?(写|生成|整理|总结|改写|翻译|列出|制定|设计|规划).+",
        r"^把.+(列出来|整理出来|总结出来|改写成).+",
        r"^请生成.+(提纲|模板|计划|方案).+",
        r"^帮我写.+",
        r"^帮我把.+",
    )
    return any(re.match(pattern, normalized) for pattern in task_patterns)


def _matches_knowledge_rule(text: str) -> bool:
    normalized = text.strip()
    knowledge_patterns = (
        r"^.+(是什么|是做什么用的|有哪些|怎么配|放哪|放哪里|如何配置|怎么启动|部署方式).*$",
        r"^配置文件里.+应该填什么.*$",
        r"^.+接口.*(有哪些|是什么).*$",
        r"^Qdrant.+做什么用.*$",
    )
    knowledge_keywords = (
        "部署方式",
        "配置文件",
        "接口",
        "Qdrant",
        "怎么启动",
        "如何配置",
        "参数",
        "说明文档",
    )
    return any(keyword in normalized for keyword in knowledge_keywords) or any(
        re.match(pattern, normalized) for pattern in knowledge_patterns
    )


def _looks_like_follow_up_by_rules(messages: list[ChatMessage]) -> bool:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not latest_user_message:
        return False

    prior_messages = [
        message
        for message in messages[:-1]
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    if not prior_messages:
        return False

    normalized = latest_user_message.replace(" ", "")
    follow_up_markers = (
        "那",
        "这个",
        "那个",
        "这边",
        "那边",
        "刚才",
        "上一个",
        "上面",
        "继续",
        "然后",
        "还有吗",
        "详细说说",
        "展开说说",
        "放哪",
        "怎么配",
        "怎么启动",
        "为什么",
    )
    if any(normalized.startswith(marker) for marker in follow_up_markers):
        return True

    return _looks_like_follow_up_fragment(latest_user_message)


def _follow_up_needs_rag(expanded_query: str, messages: list[ChatMessage]) -> bool:
    rag_keywords = (
        "文档",
        "知识库",
        "部署",
        "说明",
        "配置",
        "接口",
        "启动",
        "参数",
        "qdrant",
        "docker",
        "windows",
        "linux",
    )
    previous_user = ""
    for message in reversed(messages[:-1]):
        if message.role == "user" and message.content.strip():
            previous_user = message.content.strip()
            break

    haystack = f"{expanded_query}\n{previous_user}".lower()
    return any(keyword in haystack for keyword in rag_keywords)


def _build_task_or_qa_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    return [
        {
            "role": "system",
            "content": (
                "你是意图判断器，仅输出 task 或 knowledge_qa。\n"
                "task：要求你产出结果、编写内容、整理结构或执行任务。\n"
                "knowledge_qa：只是在询问信息、解释、配置、步骤或事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                "示例1：帮我写一个 FastAPI 健康检查接口。\n输出：task\n\n"
                "示例2：这个项目的部署方式是什么？\n输出：knowledge_qa\n\n"
                f"用户输入：{latest_user_message}\n输出："
            ),
        },
    ]


def _parse_task_or_qa_label(text: str) -> str | None:
    normalized = text.strip().lower()
    if "knowledge_qa" in normalized:
        return "knowledge_qa"
    if re.search(r"\btask\b", normalized):
        return "task"
    return None


def _normalize_knowledge_query(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact or text
