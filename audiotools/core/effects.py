import torchaudio
import torch
import numpy as np
import numbers
from . import util

def _ensure_tensor(x):
    if isinstance(x, (float, int, numbers.Integral)):
        x = np.array([x])
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    return x

def octave_bands(sample_rate, fc=1000, div=1, start=0.0, n=8):
    """
    Create a bank of octave bands
    Parameters
    ----------
    fc : float, optional
        The center frequency
    start : float, optional
        Starting frequency for octave bands in Hz (default 0.)
    n : int, optional
        Number of frequency bands (default 8)
    """
    # Octave Bands
    fcentre = fc * (
        2.0 ** (np.arange(start * div, (start + n) * div - (div - 1)) / div)
    )
    fd = 2 ** (0.5 / div)
    bands = [[f / fd, f * fd] for f in fcentre if f * fd < sample_rate / 2]
    bands.insert(0, [0, bands[0][0]])
    bands.append([bands[-1][-1], sample_rate / 2])
    return np.array(bands)

class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20
    CODEC_PRESETS = {
        "8-bit": {"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8},
        "GSM-FR": {"format": "gsm"},
        "MP3": {"format": "mp3", "compression": -9},
        "Vorbis": {"format": "vorbis", "compression": -1},
        "Ogg": {"format": "ogg", "compression": -1,},
        "Amr-nb": {"format": "amr-nb"}
    }

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
        self.audio_data = self.audio_data + other.audio_data
        return self

    def convolve(self, other, start_at_max=True):
        """
        Convolves signal one with signal two. There are three
        cases:
        
        1. s1 is multichannel and s2 is mono.
        -> s1's channels will all be convolved with s2.
        2. s1 is mono and s2 is multichannel.
        -> s1 will be convolved with each channel of s2.
        3. s1 and s2 are both multichannel.
        -> each channel will be convolved with the matching 
            channel. If they don't have the same number of
            channels, an error will be thrown.

        This function uses FFTs to do the convolution.
        """
        from .audio_signal import AudioSignal

        if start_at_max:
            idx = other.audio_data.abs().argmax(axis=-1)
            weights = [
                AudioSignal(
                    audio_array=other.audio_data[i, ..., idx[i]:],
                    sample_rate=other.sample_rate
                ) 
                for i in range(other.batch_size)
            ]
            other = AudioSignal.batch(weights, pad_signals=True)
        
        pad_len = self.signal_length - other.signal_length
        if pad_len > 0:
            other.zero_pad(0, pad_len)
        else:
            self.zero_pad(0, -pad_len-1)
        
        other.audio_data /= torch.norm(other.audio_data, p=2, dim=-1, keepdim=True)
        other_fft = torch.fft.rfft(other.audio_data)
        self_fft = torch.fft.rfft(self.audio_data)

        convolved_fft = other_fft * self_fft
        convolved_audio = torch.fft.irfft(convolved_fft)
        self.audio_data = convolved_audio

        if pad_len < 0:
            self.trim(0, -pad_len-1)
        return self

    def normalize(self, db=-24.0):
        db = _ensure_tensor(db)
        audio_signal_loudness = self.loudness()
        gain = db - audio_signal_loudness
        gain = torch.exp(gain * self.GAIN_FACTOR)
        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def _to_2d(self):
        waveform = self.audio_data.reshape(-1, self.signal_length)
        return waveform

    def _to_3d(self, waveform):
        return waveform.reshape(self.batch_size, self.num_channels, -1)

    def pitch_shift(self, n_semitones): # pragma: no cover
        effects = [
            ['pitch', str(n_semitones * 100)],
            ['rate', str(self.sample_rate)],
        ]

        waveform = self._to_2d()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self

    def time_stretch(self, factor):  # pragma: no cover
        effects = [
            ['tempo', str(factor)],
            ['rate', str(self.sample_rate)],
        ]

        waveform = self._to_2d()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self

    def apply_codec(self, preset=None, format="wav", encoding=None, 
                    bits_per_sample=None, compression=None):
        kwargs = {
            'format': format,
            'encoding': encoding,
            'bits_per_sample': bits_per_sample,
            'compression': compression
        }

        if preset is not None:
            if preset in self.CODEC_PRESETS:
                kwargs = self.CODEC_PRESETS[preset]
            else:
                raise ValueError(
                    f"Unknown preset: {preset}. "
                    f"Known presets: {list(self.CODEC_PRESETS.keys())}"
                )
        
        waveform = self._to_2d()
        if kwargs['format'] in ['vorbis', 'mp3', 'ogg', 'amr-nb']:
            # Apply it in a for loop
            augmented = torch.cat([
                torchaudio.functional.apply_codec(
                    waveform[i][None, :], self.sample_rate, **kwargs
            ) for i in range(waveform.shape[0])], dim=0)
        else:
            augmented = torchaudio.functional.apply_codec(
                waveform, self.sample_rate, **kwargs
            )
        augmented = self._to_3d(augmented)

        self.audio_data = augmented
        return self

    def octave_filterbank(self, fc=1000, start=0, div=1, n=8):
        bands = octave_bands(self.sample_rate, fc=fc, start=start, n=n)
        bands = torch.from_numpy(bands)
        closest_bins = util.hz_to_bin(bands, self.signal_length, self.sample_rate)

        fft_data = torch.fft.rfft(self.audio_data)
        filters = torch.zeros(
            *(fft_data.shape + (closest_bins.shape[0],)), 
            device=fft_data.device
        )
        for i, band in enumerate(closest_bins):
            filters[..., band[0]:band[1], i] = 1
        filtered_fft = fft_data[..., None] * filters

        fbank = torch.fft.irfft(filtered_fft.permute(0, 1, 3, 2))
        fbank = fbank.permute(0, 1, 3, 2)
        return fbank

    def equalizer(self, db, div=1):
        fbank = self.octave_filterbank(div=div)

    def __matmul__(self, other):
        return self.convolve(other)
