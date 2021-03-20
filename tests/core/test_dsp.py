from audiotools import AudioSignal
import torch
import numpy as np
import pytest

@pytest.mark.parametrize("window_duration", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100])
@pytest.mark.parametrize("duration", [0.5, 1.0, 2.0, 10.0])
def test_overlap_add(duration, sample_rate, window_duration):
    np.random.seed(0)
    if duration > window_duration:
        spk_signal = AudioSignal.batch([
            AudioSignal.excerpt('tests/audio/spk/f10_script4_produced.wav', duration=duration)
            for _ in range(16)
        ])
        spk_signal.resample(sample_rate)

        noise = torch.randn(16, 1, int(duration * sample_rate))
        nz_signal = AudioSignal(noise, sample_rate=sample_rate)

        def _test(signal):
            hop_duration = window_duration / 2

            windowed_signal = signal.deepcopy().collect_windows(
                window_duration, hop_duration)
            recombined = windowed_signal.overlap_and_add(hop_duration)

            assert recombined == signal
            assert np.allclose(recombined.audio_data, signal.audio_data, 1e-3)

        _test(nz_signal)
        _test(spk_signal)

def test_low_pass():
    sample_rate = 44100
    f = 440
    t = torch.arange(0, 1, 1 / sample_rate)
    sine_wave = torch.sin(2 * np.pi * f * t)
    signal = AudioSignal(
        sine_wave.unsqueeze(0),
        sample_rate=sample_rate
    )
    out = signal.deepcopy().low_pass(220)
    assert out.audio_data.abs().max() < 1e-4

    out = signal.deepcopy().low_pass(880)
    assert (out - signal).audio_data.abs().max() < 1e-3
    
def test_find_shelf():
    noise = torch.randn(1, 1, int(5 * 44100))
    batch_size = 5
    nz_signal = AudioSignal(noise, sample_rate=44100)
    nz_batch = AudioSignal.batch(
        [nz_signal for _ in range(batch_size)]
    )

    cutoffs = nz_batch.find_shelf()
    assert np.allclose(cutoffs, 44100, 1)

    cutoffs = nz_batch.deepcopy().low_pass(16000).find_shelf()
    assert np.allclose(cutoffs, 16000, 1)

    shelves = np.random.rand(batch_size, 1) * 22050
    cutoffs = nz_batch.deepcopy().low_pass(shelves).find_shelf()
    assert np.allclose(cutoffs, shelves, atol=1)
