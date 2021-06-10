import numpy as np
import pytest
import torch
import torchaudio

import audiotools
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import audio_signal


def test_sisdr():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.distance.SISDRLoss()

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, -np.inf)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity
