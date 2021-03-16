from numpy.random import sample
from audiotools import AudioSignal
import numpy as np

def test_play():
    array = np.zeros((1, 100))
    AudioSignal(audio_array=array, sample_rate=16000).play()

def test_embed():
    array = np.zeros((1, 100))
    AudioSignal(audio_array=array, sample_rate=16000).embed()

    AudioSignal(audio_array=array, sample_rate=16000).embed(ext='.wav')