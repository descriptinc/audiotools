import torchaudio
import torch
from audiotools import AudioSignal
import tempfile
import numpy as np
import copy
import pytest

import audiotools

def test_io():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    signal = AudioSignal(audio_path)

    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        signal.write(f.name)
        signal_from_file = AudioSignal(f.name)

    assert signal == signal_from_file
    print(signal)

    array = np.random.randn(2, 16000)
    signal = AudioSignal(audio_array=array, sample_rate=16000)

    assert np.allclose(signal.numpy().audio_data, array)

    with pytest.raises(ValueError):
        signal = AudioSignal(audio_path=audio_path, audio_array=array, sample_rate=16000)

    with pytest.raises(ValueError):
        signal = AudioSignal()

    signal = AudioSignal(audio_path, offset=10, duration=10)
    assert np.allclose(signal.signal_duration, 10.0)

def test_arithmetic():
    def _make_signals():
        array = np.random.randn(2, 16000)
        sig1 = AudioSignal(audio_array=array, sample_rate=16000)

        array = np.random.randn(2, 16000)
        sig2 = AudioSignal(audio_array=array, sample_rate=16000)
        return sig1, sig2

    # Addition (with a copy)
    sig1, sig2 = _make_signals()
    sig3 = sig1 + sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data + sig2.audio_data)

    # Addition (rmul)
    sig1, _ = _make_signals()
    sig3 = 5.0 + sig1
    assert torch.allclose(sig3.audio_data, sig1.audio_data + 5.0)

    # In place addition
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 += sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data + sig2.audio_data)

    # Subtraction (with a copy)
    sig1, sig2 = _make_signals()
    sig3 = sig1 - sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data - sig2.audio_data)

    # In place subtraction
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 -= sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data - sig2.audio_data)

    # Multiplication (element-wise)
    sig1, sig2 = _make_signals()
    sig3 = sig1 * sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data * sig2.audio_data)

    # Multiplication (gain)
    sig1, _ = _make_signals()
    sig3 = sig1 * 5.0
    assert torch.allclose(sig3.audio_data, sig1.audio_data * 5.0)

    # Multiplication (rmul)
    sig1, _ = _make_signals()
    sig3 = 5.0 * sig1
    assert torch.allclose(sig3.audio_data, sig1.audio_data * 5.0)

    # Multiplication (in-place)
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 *= sig2
    assert torch.allclose(sig3.audio_data, sig1.audio_data * sig2.audio_data)

def test_equality():
    array = np.random.randn(2, 16000)
    sig1 = AudioSignal(audio_array=array, sample_rate=16000)
    sig2 = AudioSignal(audio_array=array, sample_rate=16000)

    assert sig1 == sig2

    array = np.random.randn(2, 16000)
    sig3 = AudioSignal(audio_array=array, sample_rate=16000)

    assert sig1 != sig3

    assert sig1.numpy() != sig3.numpy()

def test_copy():
    array = np.random.randn(2, 16000)
    sig1 = AudioSignal(audio_array=array, sample_rate=16000)

    assert sig1 == sig1.copy()
    assert sig1 == sig1.deepcopy()

def test_to_from_ops():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    signal = AudioSignal(audio_path)
    signal = signal.to('cpu')
    assert signal.audio_data.device == torch.device('cpu')

    signal = signal.numpy()
    assert isinstance(signal.audio_data, np.ndarray)

    signal = signal.to()
    assert torch.is_tensor(signal.audio_data)

@pytest.mark.parametrize("window_length", [2048, 512])
@pytest.mark.parametrize("hop_length", [512, 128])
@pytest.mark.parametrize("window_type", ["sqrt_hann", "hanning", None])
def test_stft(window_length, hop_length, window_type):
    if hop_length >= window_length:
        hop_length = window_length // 2
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    stft_params = audiotools.STFTParams(
        window_length=window_length, hop_length=hop_length, window_type=window_type
    )
    for _stft_params in [None, stft_params]:
        signal = AudioSignal(audio_path, duration=10, stft_params=_stft_params)
        with pytest.raises(RuntimeError):
            signal.istft()
        
        stft_data = signal.stft()

        assert torch.allclose(signal.stft_data, stft_data)
        copied_signal = signal.deepcopy()
        copied_signal.stft()
        copied_signal = copied_signal.istft()

        assert copied_signal == signal
