import numpy as np

from audiotools import AudioSignal
from audiotools import metrics


def test_multiscale_stft():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.spectral.MultiScaleSTFTLoss()

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, 0)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity

    # Using SI-SDR Loss
    y = x.deepcopy()
    loss = metrics.spectral.MultiScaleSTFTLoss(loss_fn=metrics.distance.SISDRLoss())

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, -np.inf)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity


def test_mel_spectrogram_loss():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.spectral.MelSpectrogramLoss()

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, 0)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity


def test_phase_loss():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"

    x = AudioSignal.excerpt(audio_path, duration=1)
    y = x.deepcopy()

    loss = metrics.spectral.PhaseLoss()

    loss_val_identity = loss(x, y)
    assert np.allclose(loss_val_identity, 0)

    y = AudioSignal.excerpt(audio_path, duration=1)

    loss_val_diff = loss(x, y)
    assert loss_val_diff > loss_val_identity
