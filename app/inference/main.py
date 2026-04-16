from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.inference.backends import (
    GenerationInferenceBackend,
    IntentInferenceBackend,
    LocalHFGenerationBackend,
    MockInferenceBackend,
    MockIntentInferenceBackend,
    LocalHFIntentBackend,
)
from app.schemas.inference import (
    InferenceGenerateRequest,
    InferenceGenerateResponse,
    InferenceHealthResponse,
    InferenceIntentRequest,
    InferenceIntentResponse,
)


@dataclass
class InferenceContainer:
    settings: Settings
    generation_backend: GenerationInferenceBackend
    intent_backend: IntentInferenceBackend

    async def start(self) -> None:
        await self.generation_backend.start()
        await self.intent_backend.start()

    async def stop(self) -> None:
        await self.intent_backend.stop()
        await self.generation_backend.stop()


def build_inference_container(settings: Settings) -> InferenceContainer:
    if settings.inference_runtime_mode == "local_hf":
        generation_backend: GenerationInferenceBackend = LocalHFGenerationBackend(
            model_path=settings.llm_model_path,
            max_input_tokens=settings.llm_max_input_tokens,
            default_max_new_tokens=settings.llm_max_new_tokens,
        )
        intent_backend: IntentInferenceBackend = LocalHFIntentBackend(
            model_path=settings.intent_model_path,
            max_input_tokens=settings.llm_max_input_tokens,
        )
    else:
        generation_backend = MockInferenceBackend()
        intent_backend = MockIntentInferenceBackend()

    return InferenceContainer(
        settings=settings,
        generation_backend=generation_backend,
        intent_backend=intent_backend,
    )


def get_inference_container() -> InferenceContainer:
    return app.state.container


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_inference_container(settings)
    app.state.container = container
    await container.start()
    try:
        yield
    finally:
        await container.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    inference_app = FastAPI(
        title=f"{settings.app_name} Inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    @inference_app.get("/health", response_model=InferenceHealthResponse)
    async def health(
        container: InferenceContainer = Depends(get_inference_container),
    ) -> InferenceHealthResponse:
        return InferenceHealthResponse(
            status="ok",
            runtime_mode=container.settings.inference_runtime_mode,
            model_loaded=container.generation_backend.model_loaded,
            model_name=container.generation_backend.model_name,
            intent_model_loaded=container.intent_backend.model_loaded,
            intent_model_name=container.intent_backend.model_name,
        )

    @inference_app.post("/generate", response_model=InferenceGenerateResponse)
    async def generate(
        request: InferenceGenerateRequest,
        container: InferenceContainer = Depends(get_inference_container),
    ) -> InferenceGenerateResponse:
        answer = await container.generation_backend.generate(request)
        return InferenceGenerateResponse(
            answer=answer,
            model_name=container.generation_backend.model_name,
        )

    @inference_app.post("/intent", response_model=InferenceIntentResponse)
    async def intent(
        request: InferenceIntentRequest,
        container: InferenceContainer = Depends(get_inference_container),
    ) -> InferenceIntentResponse:
        decision, raw_output = await container.intent_backend.decide(request)
        return InferenceIntentResponse(
            decision=decision,
            model_name=container.intent_backend.model_name,
            raw_output=raw_output,
        )

    return inference_app


app = create_app()
