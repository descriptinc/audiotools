import numpy as np
import pyloudnorm
import pytest

from audiotools import AudioSignal
from audiotools import Meter


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_ffmpeg_resample(sample_rate):
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.ffmpeg_resample(sample_rate)
    assert signal.sample_rate == sample_rate
    assert signal.signal_length == sample_rate


def test_ffmpeg_loudness():
    np.random.seed(0)
    array = np.random.randn(16, 2, 16000)
    array /= np.abs(array).max()

    gains = np.random.rand(array.shape[0])[:, None, None]
    array = array * gains

    meter = pyloudnorm.Meter(16000)
    py_loudness = [meter.integrated_loudness(array[i].T) for i in range(array.shape[0])]

    ffmpeg_loudness_iso = AudioSignal(array, 16000).ffmpeg_loudness()
    assert np.allclose(py_loudness, ffmpeg_loudness_iso, atol=1)
