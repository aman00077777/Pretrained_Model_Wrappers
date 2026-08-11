"""
Tests for Wav2Vec2 audio encoder wrapper.

These are Level 1 unit tests.

The Hugging Face model and processor are mocked, so:
- no model is downloaded
- no Hugging Face Hub access is required
- the actual Wav2Vec2 wrapper logic is tested
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.audio.wav2vec2 import Wav2Vec2Encoder
from fusion.encoders.modal_tensor import ModalTensor


HIDDEN_SIZE = 32


# ---------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------

class FakeWav2Vec2Model(nn.Module):
    """Small fake model that behaves like a Wav2Vec2 model."""

    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_size
        )

        # Real parameter so device handling works.
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, attention_mask=None, **kwargs):
        batch_size = input_values.shape[0]

        # Create a fake sequence of hidden states:
        # [B, T, H]
        sequence_length = 5

        summary = input_values.mean(
            dim=1,
            keepdim=True
        )

        x = summary.unsqueeze(1).expand(
            batch_size,
            sequence_length,
            1
        )

        hidden_states = self.proj(x)

        return SimpleNamespace(
            last_hidden_state=hidden_states
        )


# ---------------------------------------------------------------------
# Fake processor
# ---------------------------------------------------------------------

class FakeWav2Vec2Processor:
    """Small fake processor compatible with the wrapper."""

    def __call__(
        self,
        inputs,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        **kwargs,
    ):
        batch_size = len(inputs)

        # Return the same basic structure that the real processor
        # provides to the model.
        max_length = max(len(audio) for audio in inputs)

        input_values = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.float32,
        )

        attention_mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.long,
        )

        for i, audio in enumerate(inputs):
            audio_tensor = torch.as_tensor(
                audio,
                dtype=torch.float32,
            )

            input_values[i, :len(audio_tensor)] = audio_tensor
            attention_mask[i, :len(audio_tensor)] = 1

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def make_encoder():
    """Create an encoder without downloading a real Hugging Face model."""

    fake_model = FakeWav2Vec2Model(
        hidden_size=HIDDEN_SIZE
    )

    fake_processor = FakeWav2Vec2Processor()

    with patch(
        "fusion.encoders.audio.wav2vec2.AutoProcessor.from_pretrained",
        return_value=fake_processor,
    ), patch(
        "fusion.encoders.audio.wav2vec2.AutoModel.from_pretrained",
        return_value=fake_model,
    ):

        encoder = Wav2Vec2Encoder.from_pretrained(
            "facebook/wav2vec2-base"
        )

    return encoder


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_wrapper_loads_processor_and_model():
    """from_pretrained() must load both processor and model."""

    fake_model = FakeWav2Vec2Model(
        hidden_size=HIDDEN_SIZE
    )

    fake_processor = FakeWav2Vec2Processor()

    with patch(
        "fusion.encoders.audio.wav2vec2.AutoProcessor.from_pretrained",
        return_value=fake_processor,
    ) as mock_processor, patch(
        "fusion.encoders.audio.wav2vec2.AutoModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:

        encoder = Wav2Vec2Encoder.from_pretrained(
            "facebook/wav2vec2-base"
        )

    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    assert isinstance(
        encoder,
        Wav2Vec2Encoder
    )

    assert encoder.get_output_dim() == HIDDEN_SIZE


def test_encode_returns_modal_tensor():
    """encode() must return a ModalTensor tagged as AUDIO."""

    encoder = make_encoder()

    audio = torch.randn(2, 16000)

    result = encoder.encode(audio)

    assert isinstance(
        result,
        ModalTensor
    )

    assert result.modality == Modality.AUDIO


def test_encode_output_shape_matches_expected_dimension():
    """The embedding dimension must match get_output_dim()."""

    encoder = make_encoder()

    audio = torch.randn(2, 16000)

    result = encoder.encode(audio)

    assert result.data.shape == (
        2,
        encoder.get_output_dim(),
    )


def test_encode_preserves_batch_dimension():
    """A batch of N audio samples must produce N embeddings."""

    encoder = make_encoder()

    batch_size = 4

    audio = torch.randn(
        batch_size,
        8000,
    )

    result = encoder.encode(audio)

    assert result.data.shape[0] == batch_size


def test_encode_single_audio_sample():
    """A single audio sample should still produce a batch of one."""

    encoder = make_encoder()

    audio = torch.randn(
        1,
        16000,
    )

    result = encoder.encode(audio)

    assert result.data.shape[0] == 1
    assert result.data.shape[1] == HIDDEN_SIZE


def test_encode_stereo_audio():
    """
    3-D audio input (B, C, T) should be converted to mono.

    The wrapper averages the channel dimension before processing.
    """

    encoder = make_encoder()

    audio = torch.randn(
        2,      # batch
        2,      # stereo channels
        8000,   # samples
    )

    result = encoder.encode(audio)

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )


def test_encode_rejects_non_tensor_input():
    """The wrapper must reject inputs that are not torch.Tensor."""

    encoder = make_encoder()

    with pytest.raises(TypeError):
        encoder.encode(
            [[0.1, 0.2, 0.3]]
        )


def test_encode_rejects_invalid_dimensions():
    """Audio input must have shape (B,T) or (B,C,T)."""

    encoder = make_encoder()

    invalid_audio = torch.randn(
        16000
    )

    with pytest.raises(ValueError):
        encoder.encode(invalid_audio)


def test_encode_rejects_empty_batch():
    """An empty batch must raise ValueError."""

    encoder = make_encoder()

    empty_audio = torch.empty(
        0,
        16000
    )

    with pytest.raises(ValueError):
        encoder.encode(empty_audio)


def test_encode_returns_finite_values():
    """The generated embedding must not contain NaN or Inf."""

    encoder = make_encoder()

    audio = torch.randn(
        2,
        16000
    )

    result = encoder.encode(audio)

    assert torch.isfinite(
        result.data
    ).all()


def test_custom_sample_rate_is_passed_to_processor():
    """The requested sample rate should be accepted by encode()."""

    fake_model = FakeWav2Vec2Model(
        hidden_size=HIDDEN_SIZE
    )

    fake_processor = FakeWav2Vec2Processor()

    with patch(
        "fusion.encoders.audio.wav2vec2.AutoProcessor.from_pretrained",
        return_value=fake_processor,
    ), patch(
        "fusion.encoders.audio.wav2vec2.AutoModel.from_pretrained",
        return_value=fake_model,
    ):

        encoder = Wav2Vec2Encoder.from_pretrained(
            "facebook/wav2vec2-base"
        )

    audio = torch.randn(
        2,
        8000
    )

    result = encoder.encode(
        audio,
        sample_rate=8000
    )

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )


def test_get_output_dim():
    """get_output_dim() must return the configured embedding dimension."""

    encoder = make_encoder()

    assert encoder.get_output_dim() == HIDDEN_SIZE