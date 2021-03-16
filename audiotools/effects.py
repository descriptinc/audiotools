import torchaudio
import torch
import numpy as np
import numbers

def _ensure_tensor(x):
    if isinstance(x, (float, int, numbers.Integral)):
        x = np.array([x])
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    return x

class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20

    def mix(self, other, snr=10): 
        """
        Mixes noise with signal at specified 
        signal-to-noise ratio.
        """
        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)        
        tgt_loudness = self.loudness() - snr
        other = other.normalize(tgt_loudness)
        self += other
        return self

    def convolve(self, other):  # pragma: no cover
        pass

    def normalize(self, db=-24.0):
        db = _ensure_tensor(db)
        audio_signal_loudness = self.loudness()
        gain = db - audio_signal_loudness
        gain = torch.exp(gain * self.GAIN_FACTOR)
        self.audio_data *= gain[:, None, None]
        return self

    def pitch_shift(self, n_semitones): # pragma: no cover
        pass

    def time_stretch(self, factor):  # pragma: no cover
        pass

    def apply_codec(self, codec):  # pragma: no cover
        pass

    def match_duration(self, duration):  # pragma: no cover
        n_samples = int(duration * self.sample_rate)
        pad_len = n_samples - self.signal_length
        if pad_len > 0:
            self.zero_pad(0, pad_len)
        return self.truncate_samples(n_samples)
