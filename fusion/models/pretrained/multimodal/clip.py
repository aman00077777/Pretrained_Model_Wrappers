"""
clip.py

Defines CLIPWrapper: wraps HuggingFace's CLIP model (vision + text towers)
as a FUSION multimodal encoder.

This is the reference implementation other multimodal wrappers in this repo
(e.g. AST for audio) follow the same BaseEncoder pattern from.

Expected config:
    {"model_name": "openai/clip-vit-base-patch32"}

The model_name is optional and defaults to:
    "openai/clip-vit-base-patch32"

Expected usage:
    wrapper = CLIPWrapper({"model_name": "openai/clip-vit-base-patch32"})

    # Or, via the shared BaseEncoder.from_config():
    wrapper = CLIPWrapper.from_config(
        {"model_name": "openai/clip-vit-base-patch32"}
    )

    image_embeds = wrapper.encode_image([pil_image_1, pil_image_2])
    text_embeds = wrapper.encode_text(
        ["a photo of a cat", "a photo of a dog"]
    )
    similarity = wrapper.compute_similarity(
        [pil_image_1],
        ["a photo of a cat"]
    )
"""

from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("clip")
class CLIPWrapper(BaseEncoder):
    """FUSION-native wrapper around HuggingFace's CLIPModel."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Dictionary optionally containing "model_name",
                a HuggingFace CLIP checkpoint ID.

                Defaults to:
                    "openai/clip-vit-base-patch32"

                Passed to BaseEncoder.__init__, which stores it as
                self.config. This allows CLIPWrapper.from_config(...)
                to work without additional code.
        """
        super().__init__(config)

        self.model_name = self.config.get(
            "model_name",
            "openai/clip-vit-base-patch32",
        )

        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> ModalTensor:
        """
        Encode image inputs using CLIP's vision tower.

        Args:
            images:
                A single PIL image or a list of PIL images.

        Returns:
            ModalTensor with modality=Modality.VISION containing
            the raw image embeddings of shape (B, D).
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)

        return ModalTensor(
            data=embeds,
            modality=Modality.VISION,
        )

    def encode_text(
        self,
        texts: Union[str, List[str]],
    ) -> ModalTensor:
        """
        Encode text inputs using CLIP's text tower.

        Args:
            texts:
                A single string or a list of strings.

        Returns:
            ModalTensor with modality=Modality.LANGUAGE containing
            the raw text embeddings of shape (B, D).
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            embeds = self.model.get_text_features(**inputs)

        return ModalTensor(
            data=embeds,
            modality=Modality.LANGUAGE,
        )

    def compute_similarity(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            images:
                A single PIL image or a list of PIL images.

            texts:
                A single string or a list of strings.

        Returns:
            Tensor of shape (B_images, B_texts) containing the cosine
            similarity between every image/text pair.
        """
        # Never route internal logic through self.encode().
        # Call the specific encoders directly because we already know
        # which argument is an image and which is text.
        image_embeds = self.encode_image(images).embedding
        text_embeds = self.encode_text(texts).embedding

        image_embeds = image_embeds / image_embeds.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        text_embeds = text_embeds / text_embeds.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        return image_embeds @ text_embeds.T

    def encode(self, inputs) -> ModalTensor:
        """
        BaseEncoder contract entry point.

        Dispatches to encode_image() or encode_text() based on the
        input data type.

        Args:
            inputs:
                str / List[str] for text, or
                PIL.Image.Image / torch.Tensor for images.

                A list of either type is also supported.

        Returns:
            ModalTensor returned by encode_image() or encode_text().

        Raises:
            TypeError:
                If the input type is not recognized.

        Note:
            A list of strings that are actually image file paths, e.g.
            ["cat.jpg", "dog.jpg"], cannot be distinguished from text
            based on type alone.

            Therefore, image inputs should be passed as loaded PIL.Image
            or torch.Tensor objects rather than file path strings.
        """
        item = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

        if isinstance(item, str):
            return self.encode_text(inputs)

        elif isinstance(item, (Image.Image, torch.Tensor)):
            return self.encode_image(inputs)

        else:
            raise TypeError(
                f"CLIPWrapper.encode() got an unsupported input type: "
                f"{type(item)}. "
                "Expected str / List[str] for text, or "
                "PIL.Image.Image / torch.Tensor "
                "(or a list of either) for images."
            )

    def get_output_dim(self) -> int:
        """
        Return the dimensionality of CLIP's projected embeddings.

        CLIP projects both image and text features into the same shared
        embedding space, so one dimension applies to both towers.

        Returns:
            int: CLIP projection dimension.
        """
        return self.model.config.projection_dim