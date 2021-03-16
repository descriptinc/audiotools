import torchaudio
import torch
import numpy as np
import numbers

class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20

    def snr(self, other):
        pass

    def convolve(self, other):
        pass

    def normalize(self, db=-24.0):
        if isinstance(db, (float, int, numbers.Integral)):
            db = np.array([db])
        if not torch.is_tensor(db):
            db = torch.from_numpy(db)
        audio_signal_loudness = self.loudness()
        gain = db - audio_signal_loudness
        gain = torch.exp(gain * self.GAIN_FACTOR)
        self.audio_data *= gain[:, None, None]
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
