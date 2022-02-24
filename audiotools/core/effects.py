import julius
import numpy as np
import torch
import torchaudio

from . import util


class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20
    CODEC_PRESETS = {
        "8-bit": {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
        "GSM-FR": {"format": "gsm"},
        "MP3": {"format": "mp3", "compression": -9},
        "Vorbis": {"format": "vorbis", "compression": -1},
        "Ogg": {
            "format": "ogg",
            "compression": -1,
        },
        "Amr-nb": {"format": "amr-nb"},
    }

    def mix(self, other, snr=10, other_eq=None):
        """
        Mixes noise with signal at specified
        signal-to-noise ratio. Optionally, the
        other signal can be equalized in-place.
        """
        snr = util.ensure_tensor(snr).to(self.device)

        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)
        tgt_loudness = self.loudness() - snr

        if other_eq is not None:
            other = other.equalizer(other_eq)

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
                    other.audio_data[i, ..., idx[i] :], sample_rate=other.sample_rate
                )
                for i in range(other.batch_size)
            ]
            other = AudioSignal.batch(weights, pad_signals=True)

        pad_len = self.signal_length - other.signal_length

        if pad_len > 0:
            other.zero_pad(0, pad_len)
        else:
            other.truncate_samples(self.signal_length)

        other.audio_data /= torch.norm(
            other.audio_data.clamp(min=1e-8), p=2, dim=-1, keepdim=True
        )

        other_fft = torch.fft.rfft(other.audio_data)
        self_fft = torch.fft.rfft(self.audio_data)

        convolved_fft = other_fft * self_fft
        convolved_audio = torch.fft.irfft(convolved_fft)

        self.audio_data = convolved_audio

        return self

    def apply_ir(self, ir, drr=None, ir_eq=None):
        if ir_eq is not None:
            ir = ir.equalizer(ir_eq)
        if drr is not None:
            ir = ir.alter_drr(drr)

        # Augment the impulse response to simulate microphone effects
        # and with varying direct-to-reverberant ratio.
        self.convolve(ir)

        # Rescale to the input's amplitude
        max_spk = self.audio_data.abs().max(dim=-1, keepdims=True).values
        max_transformed = self.audio_data.abs().max(dim=-1, keepdims=True).values
        scale_factor = max_spk / max_transformed
        self *= scale_factor

        return self

    def normalize(self, db=-24.0):
        db = util.ensure_tensor(db).to(self.device)
        ref_db = self.loudness()
        gain = db - ref_db
        gain = torch.exp(gain * self.GAIN_FACTOR)

        self.audio_data = self.audio_data * gain[:, None, None]
        self._loudness = None
        return self

    def volume_boost(self, db_boost):
        db_boost = util.ensure_tensor(db_boost).to(self.device)
        gain = torch.exp(db_boost * self.GAIN_FACTOR)
        self.audio_data = self.audio_data * gain[:, None, None]

        if self._loudness is not None:
            self._loudness += db_boost
        return self

    def _to_2d(self):
        waveform = self.audio_data.reshape(-1, self.signal_length)
        return waveform

    def _to_3d(self, waveform):
        return waveform.reshape(self.batch_size, self.num_channels, -1)

    def pitch_shift(self, n_semitones, quick=True):
        device = self.device
        effects = [
            ["pitch", str(n_semitones * 100)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def time_stretch(self, factor, quick=True):
        device = self.device
        effects = [
            ["tempo", str(factor)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def apply_codec(
        self,
        preset=None,
        format="wav",
        encoding=None,
        bits_per_sample=None,
        compression=None,
    ):  # pragma: no cover
        torchaudio_version_070 = "0.7" in torchaudio.__version__
        if torchaudio_version_070:
            return self

        kwargs = {
            "format": format,
            "encoding": encoding,
            "bits_per_sample": bits_per_sample,
            "compression": compression,
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
        if kwargs["format"] in ["vorbis", "mp3", "ogg", "amr-nb"]:
            # Apply it in a for loop
            augmented = torch.cat(
                [
                    torchaudio.functional.apply_codec(
                        waveform[i][None, :], self.sample_rate, **kwargs
                    )
                    for i in range(waveform.shape[0])
                ],
                dim=0,
            )
        else:
            augmented = torchaudio.functional.apply_codec(
                waveform, self.sample_rate, **kwargs
            )
        augmented = self._to_3d(augmented)

        self.audio_data = augmented
        return self

    def mel_filterbank(self, n_bands):
        filterbank = (
            julius.SplitBands(self.sample_rate, n_bands).float().to(self.device)
        )
        filtered = filterbank(self.audio_data)
        return filtered.permute(1, 2, 3, 0)

    def equalizer(self, db):
        if not torch.is_tensor(db):
            db = torch.from_numpy(db)
        n_bands = db.shape[-1]
        fbank = self.mel_filterbank(n_bands)

        # If there's a batch dimension, make sure it's the same.
        if db.ndim == 2:
            if db.shape[0] != 1:
                assert db.shape[0] == fbank.shape[0]
        else:
            db = db.unsqueeze(0)

        weights = (10**db).to(self.device).float()
        fbank = fbank * weights[:, None, None, :]
        eq_audio_data = fbank.sum(-1)
        self.audio_data = eq_audio_data
        return self

    def clip_distortion(self, clip_percentile):
        """Clips the signal at a given percentile. The higher it is,
        the lower the threshold for clipping.

        Parameters
        ----------
        clip_percentile : float
            Values are between 0.0 to 1.0. Typical values are 0.1 or below.

        Returns
        -------
        AudioSignal
            Audio signal with clipped audio data.
        """
        clip_percentile = util.ensure_tensor(clip_percentile)
        min_thresh = torch.quantile(self.audio_data, clip_percentile / 2)
        max_thresh = torch.quantile(self.audio_data, 1 - (clip_percentile / 2))

        if clip_percentile.ndim == 1:
            min_thresh = min_thresh[:, None, None]
            max_thresh = max_thresh[:, None, None]

        self.audio_data[self.audio_data < min_thresh] = min_thresh
        self.audio_data[self.audio_data > max_thresh] = max_thresh
        return self

    def quantization(self, quantization_channels: int):
        quantization_channels = util.ensure_tensor(quantization_channels)

        x = self.audio_data
        x = (x + 1) / 2
        x = x * quantization_channels
        x = x.floor()
        x = x / quantization_channels
        x = 2 * x - 1

        self.audio_data = x
        return self

    def mulaw_quantization(self, quantization_channels: int):
        mu = quantization_channels - 1.0
        mu = util.ensure_tensor(mu)

        x = self.audio_data

        # quantize
        x = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x = ((x + 1) / 2 * mu + 0.5).to(torch.int64)

        # unquantize
        x = (x / mu) * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu

        self.audio_data = x
        return self

    def __matmul__(self, other):
        return self.convolve(other)


class ImpulseResponseMixin:
    def decompose_ir(self):
        # Equations 1 and 2
        # -----------------
        # Breaking up into early
        # response + late field response.

        td = torch.argmax(self.audio_data, dim=-1, keepdim=True)
        t0 = int(self.sample_rate * 0.0025)

        idx = torch.arange(self.audio_data.shape[-1], device=self.device)[None, None, :]
        idx = idx.expand(self.batch_size, -1, -1)
        early_idx = (idx >= td - t0) * (idx <= td + t0)

        early_response = torch.zeros_like(self.audio_data, device=self.device)
        early_response[early_idx] = self.audio_data[early_idx]

        late_idx = ~early_idx
        late_field = torch.zeros_like(self.audio_data, device=self.device)
        late_field[late_idx] = self.audio_data[late_idx]

        # Equation 4
        # ----------
        # Decompose early response into windowed
        # direct path and windowed residual.

        window = torch.zeros_like(self.audio_data, device=self.device)
        for idx in range(self.batch_size):
            window_idx = early_idx[idx, 0].nonzero()
            window[idx, ..., window_idx] = self.get_window(
                "hanning", window_idx.shape[-1], self.device
            )
        return early_response, late_field, window

    def measure_drr(self):
        early_response, late_field, _ = self.decompose_ir()
        num = (early_response**2).sum(dim=-1)
        den = (late_field**2).sum(dim=-1)
        drr = 10 * torch.log10(num / den)
        return drr

    @staticmethod
    def solve_alpha(early_response, late_field, wd, target_drr):
        # Equation 5
        # ----------
        # Apply the good ol' quadratic formula.

        wd_sq = wd**2
        wd_sq_1 = (1 - wd) ** 2
        e_sq = early_response**2
        l_sq = late_field**2
        a = (wd_sq * e_sq).sum(dim=-1)
        b = (2 * (1 - wd) * wd * e_sq).sum(dim=-1)
        c = (wd_sq_1 * e_sq).sum(dim=-1) - torch.pow(10, target_drr / 10) * l_sq.sum(
            dim=-1
        )

        expr = ((b**2) - 4 * a * c).sqrt()
        alpha = torch.maximum(
            (-b - expr) / (2 * a),
            (-b + expr) / (2 * a),
        )
        return alpha

    def alter_drr(self, drr):
        drr = util.ensure_tensor(drr, 2, self.batch_size).to(self.device)

        early_response, late_field, window = self.decompose_ir()
        alpha = self.solve_alpha(early_response, late_field, window, drr)
        min_alpha = (
            late_field.abs().max(dim=-1)[0] / early_response.abs().max(dim=-1)[0]
        )
        alpha = torch.maximum(alpha, min_alpha)[..., None]

        aug_ir_data = (
            alpha * window * early_response
            + ((1 - window) * early_response)
            + late_field
        )
        self.audio_data = aug_ir_data
        return self
