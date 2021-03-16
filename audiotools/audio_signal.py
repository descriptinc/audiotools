import torchaudio
import numpy as np
import torch
from collections import namedtuple
from scipy import signal
import copy
from .effects import EffectMixin
from .loudness import LoudnessMixin
from . import util

STFTParams = namedtuple('STFTParams',
                        ['window_length', 'hop_length', 'window_type']
                        )
STFTParams.__new__.__defaults__ = (None,) * len(STFTParams._fields)
"""
STFTParams object is a container that holds STFT parameters - window_length, 
hop_length, and window_type. Not all parameters need to be specified. Ones that
are not specified will be inferred by the AudioSignal parameters and the settings
in `nussl.core.constants`.
"""

class AudioSignal(EffectMixin, LoudnessMixin):
    def __init__(self, audio_path=None, audio_array=None, sample_rate=44100, 
                 stft_params=None, offset=0, duration=None):
        if audio_path is None and audio_array is None:
            raise ValueError("One of audio_path or audio_array must be set!")
        if audio_path is not None and audio_array is not None:
            raise ValueError("Only one of audio_path or audio_array must be set!")

        self.path_to_input_file = None

        if audio_path is not None:
            self.load_from_file(audio_path, offset=offset, duration=duration)
        
        if audio_array is not None:
            self.load_from_array(audio_array, sample_rate)

        self.window = None
        self.stft_params = stft_params
        self.stft_data = None

        self.metadata = {}

    @classmethod
    def excerpt(cls, audio_path, duration=None, state=None, **kwargs):
        info = torchaudio.info(audio_path)
        total_duration = info.num_frames / info.sample_rate
        
        state = util.random_state(state)
        upper_bound = max(total_duration - duration, 0)
        offset = state.uniform(0, upper_bound)

        signal = cls(audio_path, offset=offset, duration=duration)
        signal.metadata['offset'] = offset
        signal.metadata['duration'] = duration

        return signal

    # I/O
    def load_from_file(self, audio_path, offset, duration, device=None):
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate

        frame_offset = min(int(sample_rate * offset), info.num_frames)
        if duration is not None:
            num_frames = min(int(sample_rate * duration), info.num_frames)
        else:
            num_frames = info.num_frames

        data, sample_rate = torchaudio.load(
            audio_path, frame_offset=frame_offset, num_frames=num_frames
        )
        self.audio_data = data
        self.sample_rate = sample_rate
        self.path_to_input_file = audio_path
        return self.to(device)

    def load_from_array(self, audio_array, sample_rate, device=None):
        self.audio_data = audio_array
        self.sample_rate = sample_rate
        return self.to(device)

    def write(self, audio_path, batch_idx=0):
        torchaudio.save(
            audio_path, self.audio_data[batch_idx], 
            self.sample_rate, bits_per_sample=32
        )
        return self

    def deepcopy(self):
        return copy.deepcopy(self)

    def copy(self):
        return copy.copy(self)

    # Signal operations
    def to_mono(self):
        self.audio_data = self.audio_data.mean(1, keepdim=True)
        return self

    # Tensor operations
    def to(self, device=None):
        if isinstance(self.audio_data, np.ndarray):
            self.audio_data = torch.from_numpy(self.audio_data)
        if device is None: device = self.device
        device = device if torch.cuda.is_available() else 'cpu'
        self.audio_data = self.audio_data.to(device).float()
        return self

    def numpy(self):
        self.audio_data = self.audio_data.detach().cpu().numpy()
        return self   

    def zero_pad(self, before, after):
        self.audio_data = torch.nn.functional.pad(
            self.audio_data, (before, after)
        )
        return self

    def truncate_samples(self, length_in_samples):
       self.audio_data =  self.audio_data[..., :length_in_samples]
       return self     

    @property
    def device(self):
        if torch.is_tensor(self.audio_data):
            return self.audio_data.device
        else:
            return 'numpy'

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
        if value.ndim < 3:
            if torch.is_tensor(value):
                value = value.unsqueeze(0)
            else:
                value = np.expand_dims(value, axis=0)
        self._audio_data = value
        self.original_signal_length = self.signal_length

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
        if window_type == 'sqrt_hann':
            window = np.sqrt(signal.get_window(
                'hann', window_length
            ))
        else:
            window = signal.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(device).float()
        return window

    @property
    def stft_params(self):
        return self._stft_params

    @stft_params.setter
    def stft_params(self, value):
        default_win_len = int(
            2 ** (np.ceil(np.log2(.032 * self.sample_rate)))
        )
        default_hop_len = default_win_len // 4
        default_win_type = 'sqrt_hann'

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type
        )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)
    
    def stft(self, window_length=None, hop_length=None, window_type=None, return_complex=True):
        """
        Computes the Short Time Fourier Transform (STFT) of :attr:`audio_data`.
        The results of the STFT calculation can be accessed from :attr:`stft_data`
        if :attr:`stft_data` is ``None`` prior to running this function or ``overwrite == True``
        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`stft_data`!
        Args:
            window_length (int): Amount of time (in samples) to do an FFT on
            hop_length (int): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str): Type of scaling to apply to the window.
            overwrite (bool): Overwrite :attr:`stft_data` with current calculation
        Returns:
            (:obj:`np.ndarray`) Calculated, complex-valued STFT from :attr:`audio_data`, 3D numpy
            array with shape `(n_frequency_bins, n_hops, n_channels)`.
        """
        self.to() # Ensure audio data is a tensor.

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length
            if hop_length is None
            else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type
            if window_type is None
            else window_type
        )

        stft_data = []

        window = self.get_window(window_type, window_length, self.audio_data.device)
        window = window.to(self.audio_data.device)

        stft_data = torch.stft(
            self.audio_data.reshape(-1, self.signal_length), 
            n_fft=window_length, hop_length=hop_length, 
            window=window, return_complex=return_complex
        )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)
        self.stft_data = stft_data
        return stft_data

    def istft(self, window_length=None, hop_length=None, window_type=None, truncate_to_length=None):
        """ Computes and returns the inverse Short Time Fourier Transform (iSTFT).
        The results of the iSTFT calculation can be accessed from :attr:`audio_data`
        if :attr:`audio_data` is ``None`` prior to running this function or ``overwrite == True``
        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`audio_data`!
        Args:
            window_length (int): Amount of time (in samples) to do an FFT on
            hop_length (int): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str): Type of scaling to apply to the window.
            overwrite (bool): Overwrite :attr:`stft_data` with current calculation
            truncate_to_length (int): truncate resultant signal to specified length. Default ``None``.
        Returns:
            (:obj:`np.ndarray`) Calculated, real-valued iSTFT from :attr:`stft_data`, 2D numpy array
            with shape `(n_channels, n_samples)`.
        """
        if self.stft_data is None:
            raise RuntimeError('Cannot do inverse STFT without self.stft_data!')

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length
            if hop_length is None
            else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type
            if window_type is None
            else window_type
        )

        window = self.get_window(window_type, window_length, self.stft_data.device)

        if truncate_to_length is None:
            truncate_to_length = self.original_signal_length
            if self.signal_length is not None:
                truncate_to_length = self.signal_length

        nb, nch, nf, nt = self.stft_data.shape
        stft_data = self.stft_data.reshape(nb * nch, nf, nt)
        audio_data = torch.istft(
            stft_data, n_fft=window_length, 
            hop_length=hop_length, window=window, 
            length=truncate_to_length
        )
        audio_data = audio_data.reshape(nb, nch, -1)
        self.audio_data = audio_data
        return self

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

    def play(self, batch_idx=0):
        from . import play_util
        play_util.play(self, batch_idx=batch_idx)
        return self

    def embed(self, batch_idx=0, ext='.mp3', display=True):
        from . import play_util
        return play_util.embed_audio(self, ext=ext, 
            display=display, batch_idx=batch_idx)

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
        dur = f'{self.signal_duration:0.3f}' if self.signal_duration else '[unknown]'
        return (
            f"{self.__class__.__name__}: "
            f"{dur} sec @ "
            f"{self.path_to_input_file if self.path_to_input_file else 'path unknown'}, "
            f"{self.sample_rate if self.sample_rate else '[unknown]'} Hz, "
            f"{self.num_channels if self.num_channels else '[unknown]'} ch, "
            f"{self.stft_params}."
        )
    
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

    def __ne__(self, other):
        return not self == other