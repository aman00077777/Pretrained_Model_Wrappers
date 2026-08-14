"""
imagebind.py

Defines ImageBindWrapper: wraps Meta's ImageBind model as a FUSION
multimodal encoder.

Follows the same BaseEncoder pattern established in clip.py, extended
to ImageBind's 3 relevant modalities (vision / language / audio):
encode_vision / encode_language / encode_audio / encode / get_output_dim.

Expected config:
    {"model_name": "imagebind_huge", "device": "cpu"}

NOTE ON encode(): unlike CLIP/ALIGN (2 modalities), ImageBind supports
3, and audio vs. text inputs can both arrive as plain strings (file
paths vs. raw text), so type alone can't always disambiguate them.
encode() handles the unambiguous cases (PIL.Image -> vision, torch.Tensor
-> vision) and falls back to requiring an explicit "modality" key for
string inputs. Flagged to Aman/Shantanu -- confirm this matches the
convention the loader expects.
"""

from typing import Any, Dict, List, Union

import torch
from PIL import Image

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("imagebind")
class ImageBindWrapper(BaseEncoder):
    """FUSION-native wrapper around Meta's ImageBind model."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.model_name = self.config.get("model_name", "imagebind_huge")
        self.device = self.config.get("device", "cpu")

        try:
            from imagebind import data
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
        except ImportError as exc:
            raise ImportError(
                "The 'imagebind' package is required for ImageBindWrapper. "
                "Install it from the official Meta repo before use."
            ) from exc

        self._data = data
        self._ModalityType = ModalityType

        if self.model_name != "imagebind_huge":
            raise ValueError(
                f"ImageBindWrapper: unsupported model_name '{self.model_name}'"
            )

        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # ImageBind projects all modalities into a shared embedding space.
        self._output_dim = 1024

    def _run(self, modality_key, tensor_inputs, modality_enum: Modality) -> ModalTensor:
        with torch.no_grad():
            embeddings = self.model({modality_key: tensor_inputs})
            pooled = embeddings[modality_key]
        return ModalTensor(data=pooled, modality=modality_enum)

    def encode_vision(
        self, images: Union[str, List[str]]
    ) -> ModalTensor:
        paths = [images] if isinstance(images, str) else list(images)
        tensor = self._data.load_and_transform_vision_data(paths, self.device)
        return self._run(self._ModalityType.VISION, tensor, Modality.VISION)

    def encode_language(self, texts: Union[str, List[str]]) -> ModalTensor:
        texts = [texts] if isinstance(texts, str) else list(texts)
        tensor = self._data.load_and_transform_text(texts, self.device)
        return self._run(self._ModalityType.TEXT, tensor, Modality.LANGUAGE)

    def encode_audio(self, audio: Union[str, List[str]]) -> ModalTensor:
        paths = [audio] if isinstance(audio, str) else list(audio)
        tensor = self._data.load_and_transform_audio_data(paths, self.device)
        return self._run(self._ModalityType.AUDIO, tensor, Modality.AUDIO)

    def encode(self, inputs) -> ModalTensor:
        """
        BaseEncoder contract entry point.

        Dispatches based on input type where unambiguous (PIL.Image ->
        vision). For string inputs (which could be either audio file
        paths or text), a dict with an explicit "modality" key is
        required: {"data": ..., "modality": "audio" | "language" | "vision"}.

        Raises:
            TypeError: If the input type/shape is not recognized.
        """
        if isinstance(inputs, dict) and "modality" in inputs:
            modality = inputs["modality"]
            data = inputs["data"]
            if modality == "vision":
                return self.encode_vision(data)
            elif modality == "language":
                return self.encode_language(data)
            elif modality == "audio":
                return self.encode_audio(data)
            else:
                raise TypeError(
                    f"ImageBindWrapper.encode() got unknown modality '{modality}'. "
                    "Expected 'vision', 'language', or 'audio'."
                )

        item = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

        if isinstance(item, (Image.Image, torch.Tensor)):
            return self.encode_vision(inputs)

        raise TypeError(
            f"ImageBindWrapper.encode() got an ambiguous or unsupported input "
            f"type: {type(item)}. String inputs (audio paths vs. text) require "
            "an explicit modality: pass {'data': ..., 'modality': 'audio'|'language'|'vision'} "
            "instead, or call encode_audio()/encode_language()/encode_vision() directly."
        )

    def get_output_dim(self) -> int:
        return self._output_dim