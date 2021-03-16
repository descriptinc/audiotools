from audiotools import AudioSignal
import torch
import numpy as np

def test_normalize():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    signal = AudioSignal(audio_path, offset=10, duration=10)
    signal = signal.normalize()
    assert np.allclose(signal.loudness(), -24, atol=1e-1)

    array = np.random.randn(1, 2, 32000)
    array = array / np.abs(array).max()

    signal = AudioSignal(audio_array=array, sample_rate=16000)
    for db_incr in np.arange(10, 75, 5):
        db = -80 + db_incr
        signal = signal.normalize(db)
        loudness = signal.loudness()
        assert np.allclose(loudness, db, atol=1e-1)

    batch_size = 16
    db = -60 + torch.linspace(10, 30, batch_size)

    array = np.random.randn(batch_size, 2, 32000)
    array = array / np.abs(array).max()
    signal = AudioSignal(audio_array=array, sample_rate=16000)

    signal = signal.normalize(db)
    assert np.allclose(signal.loudness(), db, 1e-1)    


def test_mix():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=10)

    audio_path = 'tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav'
    nz = AudioSignal(audio_path, offset=10, duration=10)

    spk.mix(nz, snr=10)
