import copy
import hashlib
import math
import pathlib
import tempfile
import warnings
from collections import namedtuple

import julius
import librosa
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

from . import util
from ..post import DiscourseMixin
from .display import DisplayMixin
from .dsp import DSPMixin
from .effects import EffectMixin
from .effects import ImpulseResponseMixin
from .ffmpeg import FFMPEGMixin
from .loudness import LoudnessMixin
from .playback import PlayMixin


STFTParams = namedtuple(
    "STFTParams", ["window_length", "hop_length", "window_type", "match_stride"]
)
STFTParams.__new__.__defaults__ = (None, None, None, None)
"""
STFTParams object is a container that holds STFT parameters - window_length,
hop_length, and window_type. Not all parameters need to be specified. Ones that
are not specified will be inferred by the AudioSignal parameters.
"""


class AudioSignal(
    EffectMixin,
    LoudnessMixin,
    PlayMixin,
    ImpulseResponseMixin,
    DSPMixin,
    DisplayMixin,
    FFMPEGMixin,
    DiscourseMixin,
):
    def __init__(
        self,
        audio_path_or_array,
        sample_rate=None,
        stft_params=None,
        offset=0,
        duration=None,
        device=None,
    ):

        audio_path = None
        audio_array = None

        if isinstance(audio_path_or_array, str):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, pathlib.Path):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, np.ndarray):
            audio_array = audio_path_or_array
        elif torch.is_tensor(audio_path_or_array):
            audio_array = audio_path_or_array
        else:
            raise ValueError(
                "audio_path_or_array must be either a Path, "
                "string, numpy array, or torch Tensor!"
            )

        self.path_to_input_file = None

        self.audio_data = None
        self.stft_data = None
        if audio_path is not None:
            self.load_from_file(
                audio_path, offset=offset, duration=duration, device=device
            )
        elif audio_array is not None:
            assert sample_rate is not None, "Must set sample rate!"
            self.load_from_array(audio_array, sample_rate, device=device)

        self.window = None
        self.stft_params = stft_params

        self.metadata = {}

    @classmethod
    def excerpt(cls, audio_path, offset=None, duration=None, state=None, **kwargs):
        info = util.info(audio_path)
        total_duration = info.num_frames / info.sample_rate

        state = util.random_state(state)
        lower_bound = 0 if offset is None else offset
        upper_bound = max(total_duration - duration, 0)
        offset = state.uniform(lower_bound, upper_bound)

        signal = cls(audio_path, offset=offset, duration=duration)
        signal.metadata["offset"] = offset
        signal.metadata["duration"] = duration

        return signal

    @classmethod
    def salient_excerpt(
        cls, audio_path, loudness_cutoff=None, num_tries=None, state=None, **kwargs
    ):
        loudness_cutoff = -np.inf if loudness_cutoff is None else loudness_cutoff
        state = util.random_state(state)
        loudness = -np.inf
        num_try = 0
        while loudness <= loudness_cutoff:
            excerpt = cls.excerpt(audio_path, state=state, **kwargs)
            loudness = excerpt.loudness()
            num_try += 1
            if num_tries is not None and num_try >= num_tries:
                break
        return excerpt

    @classmethod
    def batch(
        cls, audio_signals, pad_signals=False, truncate_signals=False, resample=False
    ):
        signal_lengths = [x.signal_length for x in audio_signals]
        sample_rates = [x.sample_rate for x in audio_signals]

        if len(set(sample_rates)) != 1:
            if resample:
                for x in audio_signals:
                    x.resample(sample_rates[0])
            else:
                raise RuntimeError(
                    f"Not all signals had the same sample rate! Got {sample_rates}. "
                    f"All signals must have the same sample rate, or resample must be True. "
                )

        if len(set(signal_lengths)) != 1:
            if pad_signals:
                max_length = max(signal_lengths)
                for x in audio_signals:
                    pad_len = max_length - x.signal_length
                    x.zero_pad(0, pad_len)
            elif truncate_signals:
                min_length = min(signal_lengths)
                for x in audio_signals:
                    x.truncate_samples(min_length)
            else:
                raise RuntimeError(
                    f"Not all signals had the same length! Got {signal_lengths}. "
                    f"All signals must be the same length, or pad_signals/truncate_signals "
                    f"must be True. "
                )
        # Concatenate along the batch dimension
        audio_data = torch.cat([x.audio_data for x in audio_signals], dim=0)
        audio_paths = [x.path_to_input_file for x in audio_signals]

        batched_signal = cls(
            audio_data,
            sample_rate=audio_signals[0].sample_rate,
        )
        batched_signal.path_to_input_file = audio_paths
        batched_signal.metadata = util.collate([x.metadata for x in audio_signals])
        return batched_signal

    # I/O
    def load_from_file(self, audio_path, offset, duration, device="cpu"):
        data, sample_rate = librosa.load(
            audio_path,
            offset=offset,
            duration=duration,
            sr=None,
            mono=False,
        )
        data = util.ensure_tensor(data)

        if data.ndim < 2:
            data = data.unsqueeze(0)
        if data.ndim < 3:
            data = data.unsqueeze(0)
        self.audio_data = data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate
        self.path_to_input_file = audio_path
        return self.to(device)

    def load_from_array(self, audio_array, sample_rate, device="cpu"):
        audio_data = util.ensure_tensor(audio_array)

        if audio_data.dtype == torch.double:
            audio_data = audio_data.float()

        if audio_data.ndim < 2:
            audio_data = audio_data.unsqueeze(0)
        if audio_data.ndim < 3:
            audio_data = audio_data.unsqueeze(0)
        self.audio_data = audio_data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate
        return self.to(device)

    def write(self, audio_path, batch_idx=0):
        if self.audio_data[batch_idx].abs().max() > 1:
            warnings.warn("Audio amplitude > 1 clipped when saving")
        torchaudio.save(audio_path, self.audio_data[batch_idx], self.sample_rate)
        return self

    def deepcopy(self):
        return copy.deepcopy(self)

    def copy(self):
        return copy.copy(self)

    def clone(self):
        clone = type(self)(
            self.audio_data.clone(),
            self.sample_rate,
            stft_params=self.stft_params,
        )
        if self.stft_data is not None:
            clone.stft_data = self.stft_data.clone()
        if self._loudness is not None:
            clone._loudness = self._loudness.clone()
        return clone

    def detach(self):
        if self._loudness is not None:
            self._loudness = self._loudness.detach()
        if self.stft_data is not None:
            self.stft_data = self.stft_data.detach()

        self.audio_data = self.audio_data.detach()
        return self

    def hash(self, batch_idx=0):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name, batch_idx)
            h = hashlib.sha256()
            b = bytearray(128 * 1024)
            mv = memoryview(b)
            with open(f.name, "rb", buffering=0) as f:
                for n in iter(lambda: f.readinto(mv), 0):
                    h.update(mv[:n])
            file_hash = h.hexdigest()
        return file_hash

    # Signal operations
    def to_mono(self):
        self.audio_data = self.audio_data.mean(1, keepdim=True)
        return self

    def resample(self, sample_rate):
        if sample_rate == self.sample_rate:
            return self
        self.audio_data = julius.resample_frac(
            self.audio_data, self.sample_rate, sample_rate
        )
        self.sample_rate = sample_rate
        return self

    # Tensor operations
    def to(self, device):
        if self._loudness is not None:
            self._loudness = self._loudness.to(device)
        if self.stft_data is not None:
            self.stft_data = self.stft_data.to(device)
        if self.audio_data is not None:
            self.audio_data = self.audio_data.to(device)
        return self

    def float(self):
        self.audio_data = self.audio_data.float()
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):  # pragma: no cover
        return self.to("cuda")

    def numpy(self):
        return self.audio_data.detach().cpu().numpy()

    def zero_pad(self, before, after):
        self.audio_data = torch.nn.functional.pad(self.audio_data, (before, after))
        return self

    def zero_pad_to(self, length, mode="after"):
        if mode == "before":
            self.zero_pad(max(length - self.signal_length, 0), 0)
        elif mode == "after":
            self.zero_pad(0, max(length - self.signal_length, 0))
        return self

    def trim(self, before, after):
        if after == 0:
            self.audio_data = self.audio_data[..., before:]
        else:
            self.audio_data = self.audio_data[..., before:-after]
        return self

    def truncate_samples(self, length_in_samples):
        self.audio_data = self.audio_data[..., :length_in_samples]
        return self

    @property
    def device(self):
        if self.audio_data is not None:
            device = self.audio_data.device
        elif self.stft_data is not None:
            device = self.stft_data.device
        return device

    # Properties
    @property
    def audio_data(self):
        return self._audio_data

    @audio_data.setter
    def audio_data(self, data):
        """Setter for audio data. Audio data is always of the shape
        (batch_size, num_channels, num_samples). If value has less
        than 3 dims (e.g. is (num_channels, num_samples)), then it will
        be reshaped to (1, num_channels, num_samples) - a batch size of 1.
        """
        if data is not None:
            assert torch.is_tensor(data), "audio_data should be torch.Tensor"
            assert data.ndim == 3, "audio_data should be 3-dim (B, C, T)"
        self._audio_data = data
        # Old loudness value not guaranteed to be right, reset it.
        self._loudness = None
        return

    @property
    def stft_data(self):
        return self._stft_data

    @stft_data.setter
    def stft_data(self, data):
        if data is not None:
            assert torch.is_tensor(data) and torch.is_complex(data)
            if self.stft_data is not None and self.stft_data.shape != data.shape:
                warnings.warn("stft_data changed shape")
        self._stft_data = data
        return

    @property
    def batch_size(self):
        return self.audio_data.shape[0]

    @property
    def signal_length(self):
        return self.audio_data.shape[-1]

    @property
    def shape(self):
        return self.audio_data.shape

    @property
    def signal_duration(self):
        return self.signal_length / self.sample_rate

    @property
    def num_channels(self):
        return self.audio_data.shape[1]

    # STFT
    @staticmethod
    def get_window(window_type, window_length, device):
        """
        Wrapper around scipy.signal.get_window so one can also get the
        popular sqrt-hann window.

        Args:
            window_type (str): Type of window to get (see constants.ALL_WINDOW).
            window_length (int): Length of the window

        Returns:
            np.ndarray: Window returned by scipy.signa.get_window
        """
        if window_type == "average":
            window = np.ones(window_length) / window_length
        elif window_type == "sqrt_hann":
            window = np.sqrt(signal.get_window("hann", window_length))
        else:
            window = signal.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(device).float()
        return window

    @property
    def stft_params(self):
        return self._stft_params

    @stft_params.setter
    def stft_params(self, value):
        default_win_len = int(2 ** (np.ceil(np.log2(0.032 * self.sample_rate))))
        default_hop_len = default_win_len // 4
        default_win_type = "sqrt_hann"
        default_match_stride = False

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type,
            match_stride=default_match_stride,
        )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)
        self.stft_data = None

    def compute_stft_padding(self, window_length, hop_length, match_stride):
        length = self.signal_length

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(length / hop_length) * hop_length - length
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        return right_pad, pad

    def stft(
        self,
        window_length=None,
        hop_length=None,
        window_type=None,
        match_stride=None,
    ):
        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length if hop_length is None else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type if window_type is None else window_type
        )
        match_stride = (
            self.stft_params.match_stride if match_stride is None else match_stride
        )

        window = self.get_window(window_type, window_length, self.audio_data.device)
        window = window.to(self.audio_data.device)

        audio_data = self.audio_data
        right_pad, pad = self.compute_stft_padding(
            window_length, hop_length, match_stride
        )
        audio_data = torch.nn.functional.pad(
            audio_data, (pad, pad + right_pad), "reflect"
        )
        stft_data = torch.stft(
            audio_data.reshape(-1, audio_data.shape[-1]),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)

        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft_data = stft_data[..., 2:-2]
        self.stft_data = stft_data

        return stft_data

    def istft(
        self,
        window_length=None,
        hop_length=None,
        window_type=None,
        match_stride=None,
        length=None,
    ):
        if self.stft_data is None:
            raise RuntimeError("Cannot do inverse STFT without self.stft_data!")

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length if hop_length is None else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type if window_type is None else window_type
        )
        match_stride = (
            self.stft_params.match_stride if match_stride is None else match_stride
        )

        window = self.get_window(window_type, window_length, self.stft_data.device)

        nb, nch, nf, nt = self.stft_data.shape
        stft_data = self.stft_data.reshape(nb * nch, nf, nt)
        right_pad, pad = self.compute_stft_padding(
            window_length, hop_length, match_stride
        )

        if length is None:
            length = self.original_signal_length
            length = length + 2 * pad + right_pad

        if match_stride:
            # Zero-pad the STFT on either side, putting back the frames that were
            # dropped in stft().
            stft_data = torch.nn.functional.pad(stft_data, (2, 2))

        audio_data = torch.istft(
            stft_data,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            length=length,
            center=True,
        )
        audio_data = audio_data.reshape(nb, nch, -1)
        if match_stride:
            audio_data = audio_data[..., pad : -(pad + right_pad)]
        self.audio_data = audio_data

        return self

    def mel_spectrogram(self, n_mels=80, mel_fmin=0.0, mel_fmax=None, **kwargs):
        stft = self.stft(**kwargs)
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = librosa_mel_fn(
            sr=self.sample_rate,
            n_fft=2 * (nf - 1),
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).to(self.device)

        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)
        return mel_spectrogram

    @property
    def magnitude(self):
        if self.stft_data is None:
            self.stft()
        return torch.abs(self.stft_data)

    @magnitude.setter
    def magnitude(self, value):
        self.stft_data = value * torch.exp(1j * self.phase)
        return

    def log_magnitude(self, ref_value=1.0, amin=1e-5, top_db=80.0):
        magnitude = self.magnitude

        amin = amin**2
        log_spec = 10.0 * torch.log10(magnitude.pow(2).clamp(min=amin))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            log_spec = torch.maximum(log_spec, log_spec.max() - top_db)
        return log_spec

    @property
    def phase(self):
        if self.stft_data is None:
            self.stft()
        return torch.angle(self.stft_data)

    @phase.setter
    def phase(self, value):
        self.stft_data = self.magnitude * torch.exp(1j * value)
        return

    # Operator overloading
    def __add__(self, other):
        new_signal = self.deepcopy()
        new_signal.audio_data += util._get_value(other)
        return new_signal

    def __iadd__(self, other):
        self.audio_data += util._get_value(other)
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        new_signal = self.deepcopy()
        new_signal.audio_data -= util._get_value(other)
        return new_signal

    def __isub__(self, other):
        self.audio_data -= util._get_value(other)
        return self

    def __mul__(self, other):
        new_signal = self.deepcopy()
        new_signal.audio_data *= util._get_value(other)
        return new_signal

    def __imul__(self, other):
        self.audio_data *= util._get_value(other)
        return self

    def __rmul__(self, other):
        return self * other

    # Representation
    def _info(self):
        dur = f"{self.signal_duration:0.3f}" if self.signal_duration else "[unknown]"
        info = {
            "duration": f"{dur} seconds",
            "batch_size": self.batch_size,
            "path": self.path_to_input_file
            if self.path_to_input_file
            else "path unknown",
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels if self.num_channels else "[unknown]",
            "audio_data.shape": self.audio_data.shape,
            "stft_params": self.stft_params,
            "device": self.device,
        }

        return info

    def markdown(self):
        info = self._info()

        FORMAT = "| Key | Value \n" "|---|--- \n"
        for k, v in info.items():
            row = f"| {k} | {v} |\n"
            FORMAT += row
        return FORMAT

    def __str__(self):
        info = self._info()

        desc = ""
        for k, v in info.items():
            desc += f"{k}: {v}\n"
        return desc

    def __rich__(self):
        from rich.table import Table

        info = self._info()

        table = Table(title=f"{self.__class__.__name__}")
        table.add_column("Key", style="green")
        table.add_column("Value", style="cyan")

        for k, v in info.items():
            table.add_row(k, str(v))
        return table

    # Comparison
    def __eq__(self, other):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                if not torch.allclose(v, other.__dict__[k], atol=1e-6):
                    max_error = (v - other.__dict__[k]).abs().max()
                    print(f"Max abs error for {k}: {max_error}")
                    return False
        return True

    # Indexing
    def __getitem__(self, key):
        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            audio_data = self.audio_data
            _loudness = self._loudness
            stft_data = self.stft_data

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            # Indexing only on the batch dimension.
            # Then let's copy over relevant stuff.
            # Future work: make this work for time-indexing
            # as well, using the hop length.
            audio_data = self.audio_data[key]
            _loudness = self._loudness[key] if self._loudness is not None else None
            stft_data = self.stft_data[key] if self.stft_data is not None else None

        copy = type(self)(audio_data, self.sample_rate, stft_params=self.stft_params)
        copy._loudness = _loudness
        copy._stft_data = stft_data

        return copy

    def __setitem__(self, key, value):
        if not isinstance(value, type(self)):
            self.audio_data[key] = value
            return

        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            self.audio_data = value.audio_data
            self._loudness = value._loudness
            self.stft_data = value.stft_data
            return

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            if self.audio_data is not None and value.audio_data is not None:
                self.audio_data[key] = value.audio_data
            if self._loudness is not None and value._loudness is not None:
                self._loudness[key] = value._loudness
            if self.stft_data is not None and value.stft_data is not None:
                self.stft_data[key] = value.stft_data
            return

    def __ne__(self, other):
        return not self == other
