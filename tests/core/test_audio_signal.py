import pathlib
import tempfile

import numpy as np
import pytest
import rich
import torch

import audiotools
from audiotools import AudioSignal


def test_io():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(pathlib.Path(audio_path))

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        signal.write(f.name)
        signal_from_file = AudioSignal(f.name)

    mp3_signal = AudioSignal(audio_path.replace("wav", "mp3"))
    print(mp3_signal)

    assert signal == signal_from_file
    print(signal)

    mp3_signal = AudioSignal.excerpt(
        audio_path.replace("wav", "mp3"), offset=5, duration=5
    )
    assert mp3_signal.signal_duration == 5.0

    rich.print(signal)

    array = np.random.randn(2, 16000)
    signal = AudioSignal(array, sample_rate=16000)
    assert np.allclose(signal.numpy().audio_data, array)

    signal = AudioSignal(array)
    assert signal.sample_rate == 44100

    with pytest.raises(ValueError):
        signal = AudioSignal(5, sample_rate=16000)

    signal = AudioSignal(audio_path, offset=10, duration=10)
    assert np.allclose(signal.signal_duration, 10.0)

    signal = AudioSignal.excerpt(audio_path, offset=5, duration=5)
    assert signal.signal_duration == 5.0

    assert "offset" in signal.metadata
    assert "duration" in signal.metadata

    signal = AudioSignal(torch.randn(1000), 44100)
    assert signal.audio_data.ndim == 3


def test_arithmetic():
    def _make_signals():
        array = np.random.randn(2, 16000)
        sig1 = AudioSignal(array, sample_rate=16000)

        array = np.random.randn(2, 16000)
        sig2 = AudioSignal(array, sample_rate=16000)
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
    sig1 = AudioSignal(array, sample_rate=16000)
    sig2 = AudioSignal(array, sample_rate=16000)

    assert sig1 == sig2

    array = np.random.randn(2, 16000)
    sig3 = AudioSignal(array, sample_rate=16000)

    assert sig1 != sig3

    assert sig1.numpy() != sig3.numpy()


def test_indexing():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    assert np.allclose(sig1[0], array[0])
    assert np.allclose(sig1[0, :, 8000], array[0, :, 8000])

    sig1[0, :, 8000] = 10
    assert np.allclose(sig1.audio_data[0, :, 8000], 10)


def test_copy():
    array = np.random.randn(2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    assert sig1 == sig1.copy()
    assert sig1 == sig1.deepcopy()


def test_zero_pad():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.zero_pad(100, 100)
    zeros = torch.zeros(4, 2, 100)
    assert torch.allclose(sig1.audio_data[..., :100], zeros)
    assert torch.allclose(sig1.audio_data[..., -100:], zeros)


def test_truncate():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.truncate_samples(100)
    assert sig1.signal_length == 100
    assert np.allclose(sig1.audio_data, array[..., :100])


def test_trim():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.trim(100, 100)
    assert sig1.signal_length == 16000 - 200
    assert np.allclose(sig1.audio_data, array[..., 100:-100])

    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.trim(0, 0)
    assert np.allclose(sig1.audio_data, array)


def test_to_from_ops():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path)
    signal = signal.to("cpu")
    assert signal.audio_data.device == torch.device("cpu")

    signal = signal.numpy()
    assert isinstance(signal.audio_data, np.ndarray)
    assert signal.device == "numpy"

    signal = signal.to()
    assert torch.is_tensor(signal.audio_data)

    signal.cpu()
    signal.cuda()
    signal.float()


@pytest.mark.parametrize("window_length", [2048, 512])
@pytest.mark.parametrize("hop_length", [512, 128])
@pytest.mark.parametrize("window_type", ["sqrt_hann", "hanning", None])
def test_stft(window_length, hop_length, window_type):
    if hop_length >= window_length:
        hop_length = window_length // 2
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
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

        mag = signal.magnitude
        phase = signal.phase

        recon_stft = mag * torch.exp(1j * phase)
        assert torch.allclose(recon_stft, signal.stft_data)

        signal.stft_data = None
        mag = signal.magnitude
        signal.stft_data = None
        phase = signal.phase

        recon_stft = mag * torch.exp(1j * phase)
        assert torch.allclose(recon_stft, signal.stft_data)


def test_to_mono():
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)
    assert signal.num_channels == 2

    signal = signal.to_mono()
    assert signal.num_channels == 1


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_resample(sample_rate):
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.resample(sample_rate)
    assert signal.sample_rate == sample_rate
    assert signal.signal_length == sample_rate


def test_batching():
    signals = []
    batch_size = 16

    # All same length, same sample rate.
    for _ in range(batch_size):
        array = np.random.randn(2, 16000)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    batched_signal = AudioSignal.batch(signals)
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, same sample rate, pad signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    max_length = max(signal_lengths)
    batched_signal = AudioSignal.batch(signals, pad_signals=True)

    assert batched_signal.signal_length == max_length
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, same sample rate, truncate signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    min_length = min(signal_lengths)
    batched_signal = AudioSignal.batch(signals, truncate_signals=True)

    assert batched_signal.signal_length == min_length
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, different sample rate, pad signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        sr = np.random.choice([8000, 16000, 32000])
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=int(sr))
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    max_length = max(signal_lengths)
    batched_signal = AudioSignal.batch(signals, resample=True, pad_signals=True)

    assert batched_signal.signal_length == max_length
    assert batched_signal.batch_size == batch_size
