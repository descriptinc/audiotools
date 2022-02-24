import torch
from numpy.random import RandomState

from audiotools import AudioSignal
from audiotools.data import transforms as tfm


def test_base_transform():
    class GainTransform(tfm.BaseTransform):
        def __init__(self, prob: float = 1.0):
            keys = ["gain"]
            super().__init__(keys=keys, prob=prob)

        def _transform(self, batch):
            signal = batch["signal"]
            gain = batch["gain"]

            signal.audio_data = signal.audio_data * gain
            batch["signal"] = signal
            return batch

        def _instantiate(self, state: RandomState, signal: AudioSignal = None):
            return {"gain": state.rand()}
