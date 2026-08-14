"""
tests/multimodal/conftest.py

Pre-stubs the missing/unavailable modules that multimodal/__init__.py tries
to import so that collection of test_blip.py and test_imagebind.py succeeds
without ModuleNotFoundError or ImportError.

Stubs installed here:
  - fusion.models.pretrained.multimodal.clip   (clip.py not yet merged)
  - fusion.models.pretrained.multimodal.align  (align.py not yet merged)
  - fusion.models.pretrained.multimodal.imagebind
        (real imagebind.py requires the 'imagebind' package which is
         optional; test_imagebind.py installs its own fake via monkeypatch
         before reloading the module)
  - imagebind / imagebind.* sub-packages       (the external package itself)

This conftest runs before pytest collects any test file in this directory.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _stub(name: str, **attrs):
    """Insert a minimal fake module into sys.modules and return it."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


# ---------------------------------------------------------------------------
# Stub the missing sibling wrappers that multimodal/__init__.py re-exports
# ---------------------------------------------------------------------------
_stub("fusion.models.pretrained.multimodal.clip", CLIPWrapper=object)
_stub("fusion.models.pretrained.multimodal.align", ALIGNWrapper=object)

# ---------------------------------------------------------------------------
# Stub the external imagebind package so imagebind.py can be imported
# without the real package installed.  test_imagebind.py will replace
# these stubs with richer fakes via monkeypatch before each test.
# ---------------------------------------------------------------------------
_ModalityType = SimpleNamespace(VISION="vision", TEXT="text", AUDIO="audio")
_fake_ib_model = MagicMock()
_fake_ib_model_mod = _stub(
    "imagebind.models.imagebind_model",
    ModalityType=_ModalityType,
    imagebind_huge=MagicMock(return_value=_fake_ib_model),
)
_stub("imagebind.models", imagebind_model=_fake_ib_model_mod)
_stub(
    "imagebind.data",
    load_and_transform_vision_data=MagicMock(return_value=MagicMock()),
    load_and_transform_text=MagicMock(return_value=MagicMock()),
    load_and_transform_audio_data=MagicMock(return_value=MagicMock()),
)
_stub("imagebind", data=sys.modules["imagebind.data"],
      models=sys.modules["imagebind.models"])
