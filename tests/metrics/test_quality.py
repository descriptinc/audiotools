import numpy as np
import pytest
import torch
import torchaudio

import audiotools
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import audio_signal


def test_stoi():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1, state=5)
    y = x.deepcopy()
    nz = AudioSignal(torch.rand_like(x.audio_data), x.sample_rate)
    nz.normalize(-24)

    loss_val_identity = metrics.quality.stoi(x, y)
    assert np.allclose(loss_val_identity, 1.0)

    y = AudioSignal.excerpt(audio_path, duration=1, state=0)

    loss_val_diff = metrics.quality.stoi(x, y)
    assert loss_val_diff < loss_val_identity

    old_stoi = 1.0
    for snr in [50, 25, 10, 5, 0, -10, -20]:
        estimate = x.deepcopy().mix(nz.deepcopy(), snr=snr)
        new_stoi = metrics.quality.stoi(estimate, x)
        assert new_stoi < old_stoi
        old_stoi = new_stoi


def test_pesq():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1, offset=5, state=5)
    y = x.deepcopy()
    nz = AudioSignal(torch.rand_like(x.audio_data), x.sample_rate)
    nz.normalize(-24)

    loss_val_identity = metrics.quality.pesq(x, y)
    assert loss_val_identity > 3.0

    y = AudioSignal.excerpt(audio_path, duration=1, offset=5, state=0)

    loss_val_diff = metrics.quality.pesq(x, y)
    assert loss_val_diff < loss_val_identity

    old_pesq = loss_val_identity
    for snr in [50, 25, 10, 0, -10, -20]:
        estimate = x.deepcopy().mix(nz.deepcopy(), snr=snr)
        new_pesq = metrics.quality.pesq(estimate, x)
        assert new_pesq < old_pesq
        old_pesq = new_pesq
