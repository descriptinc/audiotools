from numpy.random import sample
from audiotools import AudioSignal
import numpy as np

def test_specshow():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).specshow()

def test_waveplot():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).waveplot()
