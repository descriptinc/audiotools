import torchaudio
import torch

class EffectMixin:
    def snr(self, other):
        pass

    def convolve(self, other):
        pass

    def autolevel(self, db=-24):
        pass

    def pitch_shift(self, n_semitones):
        pass

    def time_stretch(self, factor):
        pass

    def apply_codec(self, codec):
        pass
