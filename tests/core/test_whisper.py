import os
import random
import tempfile

import numpy as np
import pytest
import torch

from audiotools import util
from audiotools.core.audio_signal import AudioSignal


def test_whisper_features():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=10)

    input_features = signal.get_whisper_features()

    assert input_features.dtype == torch.float32
    assert input_features.shape == (1, 80, 3000)  # (batch, channels, seq_len)


def test_whisper_transcript():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=10)

    transcript = signal.get_whisper_transcript()

    assert "<|startoftranscript|>" in transcript
    assert "<|endoftext|>" in transcript


def test_whisper_embeddings():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=10)
    embeddings = signal.get_whisper_embeddings()

    assert embeddings.dtype == torch.float32
    assert embeddings.shape == (1, 1500, 512)  # (batch, seq_len, hidden_size)
