import numpy as np
import pytest
from numpy.core.fromnumeric import clip

from audiotools import AudioSignal
from audiotools import metrics


@pytest.mark.parametrize("scaling", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("clip_min", [None, -30])
@pytest.mark.parametrize("zero_mean", [False, True])
def test_sisdr(scaling, reduction, clip_min, zero_mean):
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.distance.SISDRLoss(
        scaling=scaling, reduction=reduction, clip_min=clip_min, zero_mean=zero_mean
    )

    loss_val_identity = loss(x, y)
    lower_thresh = -np.inf if clip_min is None else clip_min
    assert np.allclose(loss_val_identity, lower_thresh)

    # Pass as tensors rather than audio signals
    loss_val_identity = loss(x.audio_data, y.audio_data)
    lower_thresh = -np.inf if clip_min is None else clip_min
    assert np.allclose(loss_val_identity, lower_thresh)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity


def test_l1_loss():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.distance.L1Loss()

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, 0.0)

    # Pass as tensors rather than audio signals
    loss_val_identity = loss(x.audio_data, y.audio_data)
    assert np.allclose(loss_val_identity, 0.0)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity
