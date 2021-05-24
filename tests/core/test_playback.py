import numpy as np
from numpy.random import sample

from audiotools import AudioSignal


def test_play():
    array = np.zeros((1, 100))
    AudioSignal(array, sample_rate=16000).play()


def test_embed():
    array = np.zeros((1, 100))
    AudioSignal(array, sample_rate=16000).embed()

    AudioSignal(array, sample_rate=16000).embed(ext=".wav")


def test_widget():
    array = np.zeros((1, 10000))
    AudioSignal(array, sample_rate=16000).widget()

    AudioSignal(array, sample_rate=16000).widget(ext=".wav")
