"""
test_clip.py

Tests for CLIPWrapper.

Requires internet access on first run because the CLIP checkpoint is
downloaded from HuggingFace.

Required packages:
    - transformers
    - torch
    - Pillow
    - pytest
"""

import torch
import pytest
from PIL import Image

from fusion.models.pretrained.multimodal.clip import CLIPWrapper


@pytest.fixture(scope="module")
def wrapper():
    return CLIPWrapper(
        {
            "model_name": "openai/clip-vit-base-patch32",
        }
    )


@pytest.fixture
def sample_images():
    """
    Create two synthetic RGB images for testing.

    If conftest.py already defines a shared sample_images fixture,
    delete this fixture and use that one instead.
    """
    red = Image.new(
        "RGB",
        (224, 224),
        color=(220, 20, 20),
    )

    blue = Image.new(
        "RGB",
        (224, 224),
        color=(20, 20, 220),
    )

    return [red, blue]


def test_clip_loads_model_and_processor(wrapper):
    """Verify that CLIP model and processor are loaded."""
    assert wrapper.model is not None
    assert wrapper.processor is not None


def test_encode_image_returns_correct_shape(wrapper, sample_images):
    """Verify image encoding returns a 2D batch of embeddings."""
    result = wrapper.encode_image(sample_images)

    assert result.embedding.ndim == 2
    assert result.embedding.shape[0] == 2


def test_encode_text_returns_correct_shape(wrapper):
    """Verify text encoding returns a 2D batch of embeddings."""
    result = wrapper.encode_text(
        [
            "a solid red square",
            "a solid blue square",
        ]
    )

    assert result.embedding.ndim == 2
    assert result.embedding.shape[0] == 2


def test_compute_similarity_output_shape_is_BxB(
    wrapper,
    sample_images,
):
    """Verify image-text similarity produces a B x B matrix."""
    texts = [
        "a solid red square",
        "a solid blue square",
    ]

    similarity = wrapper.compute_similarity(
        sample_images,
        texts,
    )

    assert similarity.shape == (2, 2)


def test_compute_similarity_embeddings_are_l2_normalized(
    wrapper,
    sample_images,
):
    """Verify image embeddings can be L2-normalized correctly."""
    image_embeds = wrapper.encode_image(sample_images).embedding

    normed = image_embeds / image_embeds.norm(
        p=2,
        dim=-1,
        keepdim=True,
    )

    norms = normed.norm(
        p=2,
        dim=-1,
    )

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-4,
    )


def test_compute_similarity_diagonal_highest_for_matching_pairs(
    wrapper,
    sample_images,
):
    """
    Verify matching image-text pairs have higher similarity.

    This is a soft smoke test using synthetic solid-color images.
    For a stricter test, use clearly distinct real photos with
    matching captions.
    """
    texts = [
        "a solid red square",
        "a solid blue square",
    ]

    similarity = wrapper.compute_similarity(
        sample_images,
        texts,
    )

    assert similarity[0, 0] >= similarity[0, 1]
    assert similarity[1, 1] >= similarity[1, 0]


def test_get_output_dim_matches_embedding_dim(
    wrapper,
    sample_images,
):
    """Verify get_output_dim() matches the actual embedding dimension."""
    result = wrapper.encode_image(sample_images)

    assert result.embedding.shape[-1] == wrapper.get_output_dim()