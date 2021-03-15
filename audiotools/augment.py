import torchaudio
import torch

class AugmentMixin:
    def snr(self, other):
        pass

    def convolve(self, other):
        pass

    def pitch_shift(self, n_semitones):
        pass

    def time_stretch(self, factor):
        pass

    def apply_codec(self, codec):
        pass
