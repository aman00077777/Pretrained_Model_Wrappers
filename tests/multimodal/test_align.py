"""
test_align.py

Tests for ALIGNWrapper.

Requires internet access on first run because the ALIGN checkpoint is
downloaded from HuggingFace.

Required packages:
    - transformers
    - torch
    - Pillow
    - pytest
"""

import pytest
from PIL import Image

from fusion.models.pretrained.multimodal.align import ALIGNWrapper
from fusion.models.pretrained.multimodal.clip import CLIPWrapper


@pytest.fixture(scope="module")
def wrapper():
    return ALIGNWrapper(
        {
            "model_name": "kakaobrain/align-base",
        }
    )


@pytest.fixture
def sample_images():
    """Create two synthetic RGB images for testing."""
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


def test_align_loads_model_and_processor(wrapper):
    """Verify that ALIGN model and processor are loaded."""
    assert wrapper.model is not None
    assert wrapper.processor is not None


def test_encode_image_returns_correct_shape(
    wrapper,
    sample_images,
):
    """Verify image encoding returns the expected shape."""
    result = wrapper.encode_image(sample_images)

    assert result.embedding.ndim == 2
    assert result.embedding.shape[0] == 2


def test_encode_text_returns_correct_shape(wrapper):
    """Verify text encoding returns the expected shape."""
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


def test_compute_similarity_matches_clip_interface_contract():
    """
    Verify that ALIGNWrapper exposes the same method surface
    as CLIPWrapper.

    ALIGNWrapper is intended to mirror the CLIPWrapper reference
    implementation.
    """
    required_methods = {
        "encode_image",
        "encode_text",
        "compute_similarity",
        "encode",
        "get_output_dim",
    }

    assert required_methods.issubset(
        set(dir(ALIGNWrapper))
    )

    assert required_methods.issubset(
        set(dir(CLIPWrapper))
    )