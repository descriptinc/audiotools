import numpy as np
from numpy.random import sample

from audiotools import AudioSignal


def test_specshow():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).specshow()


def test_waveplot():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).waveplot()


def test_wavespec():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).wavespec()
