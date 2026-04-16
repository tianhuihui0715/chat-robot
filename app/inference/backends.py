from __future__ import annotations

import re
from typing import Protocol

from app.schemas.inference import InferenceGenerateRequest


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

    async def generate(self, request: InferenceGenerateRequest) -> str:
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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    async def stop(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    async def generate(self, request: InferenceGenerateRequest) -> str:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("LocalHFGenerationBackend has not been started.")

        chat_messages = self._build_chat_messages(request)
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = self._tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if input_ids.shape[1] > self._max_input_tokens:
            input_ids = input_ids[:, -self._max_input_tokens :]
            attention_mask = attention_mask[:, -self._max_input_tokens :]

        device = self._model.device
        outputs = self._model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=request.max_new_tokens or self._default_max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_tokens = outputs[0][input_ids.shape[1] :]
        answer = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return self._sanitize_output(answer)

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

    @staticmethod
    def _sanitize_output(answer: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip() or answer.strip()
