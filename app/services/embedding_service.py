from __future__ import annotations

from threading import Lock


class SentenceTransformerProvider:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._model_path = model_path
        self._device = device
        self._model = None
        self._lock = Lock()

    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self._model_path,
                        device=self._device,
                    )
        return self._model

    def preload(self):
        return self.get_model()


class CrossEncoderProvider:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._model_path = model_path
        self._device = device
        self._model = None
        self._lock = Lock()

    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        self._model_path,
                        device=self._device,
                    )
        return self._model

    def preload(self):
        return self.get_model()
