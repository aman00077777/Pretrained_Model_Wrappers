"""
fusion — top-level package.

NOT part of the original Pretrained_Model_Wrappers repo. This package
(``fusion/``, ``fusion/exceptions.py``, ``fusion/constants.py``,
``fusion/types.py``, ``fusion/utils/``, ``fusion/encoders/base.py``,
``fusion/encoders/registry.py``) is a **stub**, added so that
``fusion.encoders.language.*`` — which ``loader.py`` already imports from
— is actually importable and testable.

``loader.py``, ``downloader.py``, and ``converter.py`` import from
``fusion.encoders.base``, ``fusion.encoders.registry``,
``fusion.exceptions``, and ``fusion.utils.logging``. None of those existed
anywhere in this repo (or in Custom_Model_Architectures) before this
change. This stub matches their contract as closely as it can be inferred
from how loader.py/converter.py actually call them — it is NOT
authoritative. If a real "fusion core" package exists elsewhere (ask
Aman Sharma), replace this whole ``fusion/`` folder with that one; the
``fusion/encoders/language/`` wrappers should need little to no change.
"""
