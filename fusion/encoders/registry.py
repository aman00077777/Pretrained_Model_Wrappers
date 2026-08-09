"""
fusion/encoders/registry.py — STUB. See fusion/__init__.py for why this exists.

Reverse-engineered from actual usage in loader.py and tests/test_loader.py:
``Registry("pretrained")``, ``@REGISTRY.register("name")`` as a decorator,
``REGISTRY.get(name)``, ``REGISTRY.list()``, ``name in REGISTRY``, and
tests reach into ``REGISTRY._store`` directly — so the internal dict is
named ``_store``, not a private-mangled or differently-named attribute.
"""

from __future__ import annotations

from typing import Callable, Dict, List


class Registry:
    """A named registry mapping string keys to zero-argument factories.

    Args:
        name (str): Human-readable name for this registry (used in errors).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: Dict[str, Callable[[], object]] = {}

    def register(self, key: str) -> Callable[[Callable], Callable]:
        """Decorator: register the decorated zero-arg factory under *key*."""

        def _decorator(factory: Callable) -> Callable:
            self._store[key] = factory
            return factory

        return _decorator

    def get(self, key: str) -> Callable[[], object]:
        """Return the factory registered under *key*."""
        return self._store[key]

    def list(self) -> List[str]:
        """Return all registered keys."""
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)
