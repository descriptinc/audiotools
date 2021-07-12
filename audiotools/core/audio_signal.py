import copy
import pathlib
from collections import namedtuple

import julius
import librosa
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

from . import util
from .display import DisplayMixin
from .dsp import DSPMixin
from .effects import EffectMixin
from .effects import ImpulseResponseMixin
from .ffmpeg import FFMPEGMixin
from .loudness import LoudnessMixin
from .playback import PlayMixin


STFTParams = namedtuple("STFTParams", ["window_length", "hop_length", "window_type"])
STFTParams.__new__.__defaults__ = (None,) * len(STFTParams._fields)
"""
STFTParams object is a container that holds STFT parameters - window_length,
hop_length, and window_type. Not all parameters need to be specified. Ones that
are not specified will be inferred by the AudioSignal parameters and the settings
in `nussl.core.constants`.
"""


class AudioSignal(
    EffectMixin,
    LoudnessMixin,
    PlayMixin,
    ImpulseResponseMixin,
    DSPMixin,
    DisplayMixin,
    FFMPEGMixin,
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

        if audio_path is not None:
            self.load_from_file(
                audio_path, offset=offset, duration=duration, device=device
            )
        elif audio_array is not None:
            self.load_from_array(audio_array, sample_rate, device=device)

        self.window = None
        self.stft_params = stft_params
        self.stft_data = None

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
        cls, audio_path, loudness_cutoff=-np.inf, num_tries=None, state=None, **kwargs
    ):
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
        audio_mask = torch.cat([x.audio_mask for x in audio_signals], dim=0)

        batched_signal = cls(
            audio_data,
            sample_rate=audio_signals[0].sample_rate,
        )
        batched_signal.audio_mask = audio_mask
        return batched_signal

    # I/O
    def load_from_file(self, audio_path, offset, duration, device=None):
        try:
            info = util.info(audio_path)
            sample_rate = info.sample_rate

            frame_offset = min(int(sample_rate * offset), info.num_frames)
            if duration is not None:
                num_frames = min(int(sample_rate * duration), info.num_frames)
            else:
                num_frames = info.num_frames

            # Compatible with torchaudio 0.7.2 and 0.8.1.
            torchaudio_version_070 = "0.7" in torchaudio.__version__
            kwargs = {
                "offset" if torchaudio_version_070 else "frame_offset": frame_offset,
                "num_frames": num_frames,
            }

            data, sample_rate = torchaudio.load(audio_path, **kwargs)
        except:  # pragma: no cover
            data, sample_rate = librosa.load(
                audio_path,
                offset=offset,
                duration=duration,
                sr=None,
            )
            data = torch.from_numpy(data)

        self.audio_data = data
        self.original_signal_length = self.signal_length

        self.audio_mask = torch.ones_like(self.audio_data)
        self.sample_rate = sample_rate
        self.path_to_input_file = audio_path
        return self.to(device)

    def load_from_array(self, audio_array, sample_rate, device=None):
        self.audio_data = audio_array
        self.original_signal_length = self.signal_length

        if self.device == "numpy":
            self.audio_mask = np.ones_like(audio_array)
        else:
            self.audio_mask = torch.ones_like(self.audio_data)
        if sample_rate is None:
            sample_rate = 44100
        self.sample_rate = sample_rate
        return self.to(device)

    def write(self, audio_path, batch_idx=0):
        torchaudio.save(audio_path, self.audio_data[batch_idx], self.sample_rate)
        return self

    def deepcopy(self):
        return copy.deepcopy(self)

    def copy(self):
        return copy.copy(self)

    # Signal operations
    def to_mono(self):
        self.audio_data = self.audio_data.mean(1, keepdim=True)
        self.audio_mask = self.audio_mask.mean(1, keepdim=True)
        return self

    def resample(self, sample_rate):
        if sample_rate == self.sample_rate:
            return self
        self.to()  # Ensure tensors.
        self.audio_data = julius.resample_frac(
            self.audio_data, self.sample_rate, sample_rate
        )
        self.audio_mask = julius.resample_frac(
            self.audio_mask, self.sample_rate, sample_rate
        )
        self.sample_rate = sample_rate
        return self

    # Tensor operations
    def to(self, device=None):
        if isinstance(self.audio_data, np.ndarray):
            self.audio_data = torch.from_numpy(self.audio_data)
        if isinstance(self.audio_mask, np.ndarray):
            self.audio_mask = torch.from_numpy(self.audio_mask)
        if device is None:
            device = self.device
        device = device if torch.cuda.is_available() else "cpu"
        self.audio_data = self.audio_data.to(device).float()
        self.audio_mask = self.audio_mask.to(device).float()
        return self

    def float(self):
        self.audio_data = self.audio_data.float()
        self.audio_mask = self.audio_mask.float()
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def numpy(self):
        self.audio_data = self.audio_data.detach().cpu().numpy()
        self.audio_mask = self.audio_mask.detach().cpu().numpy()
        return self

    def zero_pad(self, before, after):
        self.audio_data = torch.nn.functional.pad(self.audio_data, (before, after))
        self.audio_mask = torch.nn.functional.pad(self.audio_mask, (before, after))
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
            self.audio_mask = self.audio_mask[..., before:]
        else:
            self.audio_data = self.audio_data[..., before:-after]
            self.audio_mask = self.audio_mask[..., before:-after]
        return self

    def truncate_samples(self, length_in_samples):
        self.audio_data = self.audio_data[..., :length_in_samples]
        self.audio_mask = self.audio_mask[..., :length_in_samples]
        return self

    @property
    def device(self):
        if torch.is_tensor(self.audio_data):
            return self.audio_data.device
        else:
            return "numpy"

    # Properties
    @property
    def audio_data(self):
        return self._audio_data

    @audio_data.setter
    def audio_data(self, value):
        """Setter for audio data. Audio data is always of the shape
        (batch_size, num_channels, num_samples). If value has less
        than 3 dims (e.g. is (num_channels, num_samples)), then it will
        be reshaped to (1, num_channels, num_samples) - a batch size of 1.
        """
        if value.ndim < 2:
            value = value.unsqueeze(0)
        if value.ndim < 3:
            if torch.is_tensor(value):
                value = value.unsqueeze(0)
            else:
                value = np.expand_dims(value, axis=0)
        self._audio_data = value

    @property
    def batch_size(self):
        return self.audio_data.shape[0]

    @property
    def signal_length(self):
        return self.audio_data.shape[-1]

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
        if window_type == "sqrt_hann":
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

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type,
        )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)

    def stft(
        self, window_length=None, hop_length=None, window_type=None, return_complex=True
    ):
        self.to()  # Ensure audio data is a tensor.

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

        stft_data = []

        window = self.get_window(window_type, window_length, self.audio_data.device)
        window = window.to(self.audio_data.device)

        stft_data = torch.stft(
            self.audio_data.reshape(-1, self.signal_length),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=return_complex,
        )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)
        self.stft_data = stft_data
        return stft_data

    def istft(
        self,
        window_length=None,
        hop_length=None,
        window_type=None,
        truncate_to_length=None,
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

        window = self.get_window(window_type, window_length, self.stft_data.device)

        if truncate_to_length is None:
            truncate_to_length = self.original_signal_length
            if self.signal_length is not None:
                truncate_to_length = self.signal_length

        nb, nch, nf, nt = self.stft_data.shape
        stft_data = self.stft_data.reshape(nb * nch, nf, nt)
        audio_data = torch.istft(
            stft_data,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            length=truncate_to_length,
        )
        audio_data = audio_data.reshape(nb, nch, -1)
        self.audio_data = audio_data
        return self

    def mel_spectrogram(self, n_mels=80, mel_fmin=0.0, mel_fmax=None, **kwargs):
        stft = self.stft(**kwargs)
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = librosa_mel_fn(
            self.sample_rate, 2 * (nf - 1), n_mels, mel_fmin, mel_fmax
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

    @property
    def phase(self):
        if self.stft_data is None:
            self.stft()
        return torch.angle(self.stft_data)

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
    def __str__(self):
        dur = f"{self.signal_duration:0.3f}" if self.signal_duration else "[unknown]"
        return (
            f"{self.__class__.__name__}\n"
            f"Duration: {dur} sec\n"
            f"Batch size: {self.batch_size}\n"
            f"Path: {self.path_to_input_file if self.path_to_input_file else 'path unknown'}\n"
            f"Sample rate: {self.sample_rate if self.sample_rate else '[unknown]'} Hz\n"
            f"Number of channels: {self.num_channels if self.num_channels else '[unknown]'} ch\n"
            f"Audio data shape: {self.audio_data.shape}\n"
            f"STFT Parameters: {self.stft_params}"
        )

    def __rich__(self):
        from rich.table import Table

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
        }

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
            elif isinstance(v, np.ndarray):
                if not np.allclose(v, other.__dict__[k]):
                    max_error = np.abs(v - other.__dict__[k]).max()
                    print(f"Max abs error for {k}: {max_error}")
                    return False
        return True

    # Indexing
    def __getitem__(self, key):
        return self.audio_data[key]

    def __setitem__(self, key, value):
        self.audio_data[key] = value

    def __ne__(self, other):
        return not self == other
