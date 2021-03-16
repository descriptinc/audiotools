import torchaudio
import torch
import numpy as np

class EffectMixin:
    def snr(self, other):
        pass

    def convolve(self, other):
        pass

    def normalize(self, db=-24):
        audio_signal_loudness = self.loudness()
        gain = db - audio_signal_loudness
        self *= torch.exp(gain * np.log(10) / 20)
        return self

    def pitch_shift(self, n_semitones):
        pass

    def time_stretch(self, factor):
        pass

    def apply_codec(self, codec):
        pass

    def match_duration(self, duration):
        n_samples = int(duration * self.sample_rate)
        pad_len = n_samples - self.signal_length
        if pad_len > 0:
            self.zero_pad(0, pad_len)
        self.truncate_samples(n_samples)
        return self
