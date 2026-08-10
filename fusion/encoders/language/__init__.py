"""
Language encoders public API.

IMPORTANT — naming: the roadmap doc's file/class table (clip_text.py ->
CLIPTextWrapper, bert.py -> BERTWrapper, ..., gpt2.py -> GPT2Wrapper) does
NOT match what loader.py (already merged) actually imports. loader.py's
registry factories import ``BERTEncoder``, ``CLIPTextEncoder``,
``RoBERTaEncoder``, ``T5Encoder``, ``GPTEncoder`` (from ``gpt.py``, not
``gpt2.py``), and ``XLMEncoder`` — the names and gpt.py/gpt2.py filename
used below match loader.py, not the roadmap PDF, since loader.py is the
binding contract that ``load_pretrained()`` actually calls.

Exported classes:
    - CLIPTextEncoder
    - BERTEncoder
    - RoBERTaEncoder
    - T5Encoder
    - GPTEncoder
    - XLMEncoder
"""

from fusion.encoders.language.clip_text import CLIPTextEncoder
from fusion.encoders.language.bert import BERTEncoder
from fusion.encoders.language.roberta import RoBERTaEncoder
from fusion.encoders.language.t5 import T5Encoder
from fusion.encoders.language.gpt import GPTEncoder
from fusion.encoders.language.xlm import XLMEncoder

__all__ = [
    "CLIPTextEncoder",
    "BERTEncoder",
    "RoBERTaEncoder",
    "T5Encoder",
    "GPTEncoder",
    "XLMEncoder",
]
