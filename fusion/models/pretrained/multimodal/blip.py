"""
blip.py

Defines BLIPWrapper: wraps HuggingFace's BLIP model
(Salesforce/blip-image-captioning-base) as a FUSION multimodal encoder.

Follows the same BaseEncoder pattern established in clip.py:
encode_image / encode_text / encode / get_output_dim, plus
generate_caption as a BLIP-specific extra.

Expected config:
    {"model_name": "Salesforce/blip-image-captioning-base"}

The model_name is optional and defaults to:
    "Salesforce/blip-image-captioning-base"
"""

from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("blip")
class BLIPWrapper(BaseEncoder):
    """FUSION-native wrapper around HuggingFace's BlipForConditionalGeneration."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.model_name = self.config.get(
            "model_name",
            "Salesforce/blip-image-captioning-base",
        )

        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.processor = BlipProcessor.from_pretrained(self.model_name)

        self.model.eval()

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> ModalTensor:
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            embeds = vision_outputs.pooler_output

        return ModalTensor(data=embeds, modality=Modality.VISION)

    def encode_text(
        self,
        texts: Union[str, List[str]],
    ) -> ModalTensor:
        # BLIP's text side is decoder-oriented, no dedicated pooler
        # like CLIP/ALIGN -- mean-pool the last hidden state instead.
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            text_outputs = self.model.text_decoder.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            token_embeddings = text_outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeds = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return ModalTensor(data=embeds, modality=Modality.LANGUAGE)

    def generate_caption(self, image: Image.Image) -> str:
        if image is None:
            raise ValueError(
                "BLIPWrapper.generate_caption() received None; "
                "expected a PIL.Image.Image."
            )

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=30)

        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def encode(self, inputs) -> ModalTensor:
        item = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

        if isinstance(item, str):
            return self.encode_text(inputs)
        elif isinstance(item, (Image.Image, torch.Tensor)):
            return self.encode_image(inputs)
        else:
            raise TypeError(
                f"BLIPWrapper.encode() got an unsupported input type: "
                f"{type(item)}. Expected str / List[str] for text, or "
                "PIL.Image.Image / torch.Tensor (or a list of either) for images."
            )

    def get_output_dim(self) -> int:
        return self.model.config.vision_config.hidden_size