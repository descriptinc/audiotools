from subprocess import check_output
from audiotools import AudioSignal
import torch
import numpy as np
import pytest

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

    spk.deepcopy().mix(nz, snr=-10)
    snr = spk.loudness() - nz.loudness()
    assert np.allclose(snr, -10, atol=1)

    # Test in batch
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=10)

    audio_path = 'tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav'
    nz = AudioSignal(audio_path, offset=10, duration=10)

    batch_size = 4
    tgt_snr = torch.linspace(-10, 10, batch_size)

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    nz_batch = AudioSignal.batch([nz.deepcopy() for _ in range(batch_size)])

    spk_batch.deepcopy().mix(nz_batch, snr=tgt_snr)
    snr = spk_batch.loudness() - nz_batch.loudness()
    assert np.allclose(snr, tgt_snr, atol=1)

def test_convolve():
    np.random.seed(6) # Found a failing seed
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=10)

    impulse = np.zeros((1, 16000))
    impulse[..., 0] = 1
    ir = AudioSignal(audio_array=impulse)
    batch_size = 4

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    ir_batch = AudioSignal.batch(
        [ir.deepcopy().zero_pad(np.random.randint(1000), 0) for _ in range(batch_size)],
        pad_signals=True
    )

    convolved = spk_batch.deepcopy().convolve(ir_batch)
    assert convolved == spk_batch

    # Short duration
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=0.1)

    impulse = np.zeros((1, 16000))
    impulse[..., 0] = 1
    ir = AudioSignal(audio_array=impulse)
    batch_size = 4

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    ir_batch = AudioSignal.batch(
        [ir.deepcopy().zero_pad(np.random.randint(1000), 0) for _ in range(batch_size)],
        pad_signals=True
    )

    convolved = spk_batch.deepcopy().convolve(ir_batch)
    assert convolved == spk_batch

def test_pipeline():
    # An actual IR, no batching
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=5)

    audio_path = 'tests/audio/ir/h179_Bar_1txts.wav'
    ir = AudioSignal(audio_path)
    spk.deepcopy().convolve(ir)
    
    audio_path = 'tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav'
    nz = AudioSignal(audio_path, offset=10, duration=5)

    batch_size = 16
    tgt_snr = torch.linspace(20, 30, batch_size)

    (spk @ ir).mix(nz, snr=tgt_snr)

def test_codec():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=10)
    
    with pytest.raises(ValueError):
        spk.apply_codec("unknown preset")

    out = spk.deepcopy().apply_codec("Ogg")
    out = spk.deepcopy().apply_codec("8-bit")

def test_pitch_shift():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=1)
    
    out = spk.deepcopy().play().pitch_shift(5).play()

def test_time_stretch():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    spk = AudioSignal(audio_path, offset=10, duration=1)
    
    out = spk.deepcopy().play().time_stretch(0.8).play()
