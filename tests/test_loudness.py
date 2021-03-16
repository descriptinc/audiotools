from audiotools import AudioSignal
from audiotools.loudness import Meter
import pyloudnorm
import numpy as np
import soundfile as sf

def test_loudness_against_pyln():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    signal = AudioSignal(audio_path, offset=5, duration=10)
    signal_loudness = signal.loudness()

    meter = pyloudnorm.Meter(
        signal.sample_rate, 
        filter_class="K-weighting", 
        block_size=0.4
    )
    py_loudness = meter.integrated_loudness(signal.numpy().audio_data[0].T)
    assert np.allclose(signal_loudness, py_loudness, 1e-1)

def test_loudness_short():
    audio_path = 'tests/audio/spk/f10_script4_produced.wav'
    signal = AudioSignal(audio_path, offset=10, duration=0.25)
    signal_loudness = signal.loudness()

def test_batch_loudness():
    np.random.seed(0)
    array = np.random.randn(16, 2, 16000)
    array /= np.abs(array).max()

    gains = np.random.rand(array.shape[0])[:, None, None]
    array = array * gains

    meter = pyloudnorm.Meter(16000)
    py_loudness = [
        meter.integrated_loudness(array[i].T)
        for i in range(array.shape[0])
    ]

    meter = Meter(16000)
    at_loudness_iso = [
        meter.integrated_loudness(array[i].T).item()
        for i in range(array.shape[0])
    ]

    assert np.allclose(py_loudness, at_loudness_iso, atol=1e-1)

    signal = AudioSignal(audio_array=array, sample_rate=16000)
    at_loudness_batch = signal.loudness()
    assert np.allclose(py_loudness, at_loudness_batch, atol=1e-1)

# Tests below are copied from pyloudnorm
def test_integrated_loudness():
    data, rate = sf.read("tests/audio/loudness/sine_1000.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -3.0523438444331137
    assert np.allclose(loudness, targetLoudness, atol=5e-1)

def test_rel_gate_test():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_RelGateTest.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)
	
    targetLoudness = -10.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_abs_gate_test():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_AbsGateTest.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)
	
    targetLoudness = -69.5
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_24LKFS_25Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_25Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)
	
    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)
	
def test_24LKFS_100Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_100Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)
	
    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)
	
def test_24LKFS_500Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_500Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_24LKFS_1000Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_1000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_24LKFS_2000Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_2000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_24LKFS_10000Hz_2ch():
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_24LKFS_10000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_25Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_25Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_100Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_100Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_500Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_500Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_1000Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_1000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_2000Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_2000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_23LKFS_10000Hz_2ch(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_23LKFS_10000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_18LKFS_frequency_sweep(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Comp_18LKFS_FrequencySweep.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -18.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_conf_stereo_vinL_R_23LKFS(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Conf_Stereo_VinL+R-23LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_conf_monovoice_music_24LKFS(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def conf_monovoice_music_24LKFS(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)

def test_conf_monovoice_music_23LKFS(): 
    data, rate = sf.read("tests/audio/loudness/1770-2_Conf_Mono_Voice+Music-23LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=0.1)
