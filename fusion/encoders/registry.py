"""
Task H2 — Encoder Registry

Provides a generic name -> class registry, plus the single shared
ENCODER_REGISTRY instance that every vision/language/audio encoder
registers itself into (except "custom" encoders, which must NOT register).
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List

from fusion.utils.logging import get_logger

logger = get_logger(__name__)


class Registry:
    """A simple named registry mapping string keys to classes/callables."""

    def __init__(self, name: str):
        self._name = name
        self._store: Dict[str, Any] = {}
        
        
    def register(self, key: str) -> Callable:
        """Class decorator: @ENCODER_REGISTRY.register("bert")"""
        if key in self._store:
            raise ValueError(
                f"event=duplicate_registry_name registry={self._name} key={key}"
            )

        def _decorator(cls_or_fn):
            self._store[key] = cls_or_fn
            logger.debug("event=registered registry=%s key=%s", self._name, key)
            return cls_or_fn

        return _decorator

    def build(self, key: str, *args, **kwargs):
        if key not in self._store:
            raise KeyError(
                f"event=unknown_registry_key registry={self._name} key={key} "
                f"available={self.list()}"
            )
        return self._store[key](*args, **kwargs)

    def get(self, key: str):
        if key not in self._store:
            raise KeyError(f"event=unknown_registry_key registry={self._name} key={key}")
        return self._store[key]

    def list(self) -> List[str]:
        return sorted(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store


ENCODER_REGISTRY = Registry("encoders")

# Convenience decorator used by individual encoder modules:
#   @register_encoder("whisper")
#   class WhisperEncoder(BaseEncoder): ...
register_encoder = ENCODER_REGISTRY.register

__all__ = ["ENCODER_REGISTRY", "Registry", "register_encoder"]
