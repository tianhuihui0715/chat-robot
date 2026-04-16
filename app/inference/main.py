from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.inference.backends import (
    InferenceBackend,
    LocalHFGenerationBackend,
    MockInferenceBackend,
)
from app.schemas.inference import (
    InferenceGenerateRequest,
    InferenceGenerateResponse,
    InferenceHealthResponse,
)


@dataclass
class InferenceContainer:
    settings: Settings
    backend: InferenceBackend

    async def start(self) -> None:
        await self.backend.start()

    async def stop(self) -> None:
        await self.backend.stop()


def build_inference_container(settings: Settings) -> InferenceContainer:
    if settings.inference_runtime_mode == "local_hf":
        backend: InferenceBackend = LocalHFGenerationBackend(
            model_path=settings.llm_model_path,
            max_input_tokens=settings.llm_max_input_tokens,
            default_max_new_tokens=settings.llm_max_new_tokens,
        )
    else:
        backend = MockInferenceBackend()

    return InferenceContainer(settings=settings, backend=backend)


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
            model_loaded=container.backend.model_loaded,
            model_name=container.backend.model_name,
        )

    @inference_app.post("/generate", response_model=InferenceGenerateResponse)
    async def generate(
        request: InferenceGenerateRequest,
        container: InferenceContainer = Depends(get_inference_container),
    ) -> InferenceGenerateResponse:
        answer = await container.backend.generate(request)
        return InferenceGenerateResponse(
            answer=answer,
            model_name=container.backend.model_name,
        )

    return inference_app


app = create_app()
