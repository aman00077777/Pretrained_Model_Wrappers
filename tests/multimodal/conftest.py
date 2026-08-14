"""
tests/multimodal/conftest.py

Pre-stubs the external ``imagebind`` package in sys.modules so that
multimodal/__init__.py (which imports ImageBindWrapper) can be collected
without the real imagebind package being installed.

test_imagebind.py replaces these bare stubs with richer fakes via
monkeypatch before each test fixture runs.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _stub(name: str, **attrs):
    """Insert a minimal fake module into sys.modules if not already present."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


# ---------------------------------------------------------------------------
# Stub the external imagebind package so imagebind.py can be imported at
# collection time without the real package installed.
# test_imagebind.py will replace these entries with richer fakes per-test.
# ---------------------------------------------------------------------------
_ModalityType = SimpleNamespace(VISION="vision", TEXT="text", AUDIO="audio")
_fake_ib_model = MagicMock()

_ib_model_mod = _stub(
    "imagebind.models.imagebind_model",
    ModalityType=_ModalityType,
    imagebind_huge=MagicMock(return_value=_fake_ib_model),
)
_stub("imagebind.models", imagebind_model=_ib_model_mod)
_stub(
    "imagebind.data",
    load_and_transform_vision_data=MagicMock(return_value=MagicMock()),
    load_and_transform_text=MagicMock(return_value=MagicMock()),
    load_and_transform_audio_data=MagicMock(return_value=MagicMock()),
)
_stub(
    "imagebind",
    data=sys.modules["imagebind.data"],
    models=sys.modules["imagebind.models"],
)
