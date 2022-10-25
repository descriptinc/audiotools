import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm
import pytest
import torch

from audiotools import AudioSignal


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_ffmpeg_resample(sample_rate):
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.ffmpeg_resample(sample_rate)
    assert signal.sample_rate == sample_rate
    assert signal.signal_length == sample_rate


def test_ffmpeg_loudness():
    np.random.seed(0)
    array = np.random.randn(16, 2, 16000)
    array /= np.abs(array).max()

    gains = np.random.rand(array.shape[0])[:, None, None]
    array = array * gains

    meter = pyloudnorm.Meter(16000)
    py_loudness = [meter.integrated_loudness(array[i].T) for i in range(array.shape[0])]

    ffmpeg_loudness_iso = AudioSignal(array, 16000).ffmpeg_loudness()
    assert np.allclose(py_loudness, ffmpeg_loudness_iso, atol=1)

    # if you normalize and then write, it should still work.
    # if ffmpeg is float64, this fails
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        x = AudioSignal(torch.randn(44100 * 10), 44100)
        x.ffmpeg_loudness(-24)
        x.normalize(-24)
        x.write(f.name)


def test_ffmpeg_load():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    # convert to mp3 with ffmpeg
    og_signal = AudioSignal(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        command = f"ffmpeg -i {audio_path} {f.name} -y -hide_banner -loglevel error"
        subprocess.check_call(shlex.split(command))

        signal_from_ffmpeg = AudioSignal.load_from_file_with_ffmpeg(f.name)
        assert og_signal.signal_length == signal_from_ffmpeg.signal_length

    # test spaces in title
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "Title with spaces.wav")
        og_signal.write(out_path)
        signal_from_ffmpeg = AudioSignal.load_from_file_with_ffmpeg(out_path)

        assert og_signal.signal_length == signal_from_ffmpeg.signal_length

    # test quotes in title
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "Someone's title with spaces.wav")
        og_signal.write(out_path)
        signal_from_ffmpeg = AudioSignal.load_from_file_with_ffmpeg(out_path)

        assert og_signal.signal_length == signal_from_ffmpeg.signal_length


def test_ffmpeg_audio_offset():
    with tempfile.TemporaryDirectory() as d:
        video_path = Path(d) / "test.mp4"
        delayed_video = Path(d) / "test_delayed.mp4"

        # Create a test video
        subprocess.run(
            shlex.split(
                f"ffmpeg -y -f lavfi "
                f"-i testsrc=d=5:s=120x120:r=24,format=yuv420p "
                f"-f lavfi -i sine=f=440:b=4 "
                f"-shortest {video_path} -loglevel error"
            )
        )

        # Create a video with the audio offset by 1 second
        subprocess.run(
            shlex.split(
                f"ffmpeg -y -i {video_path} "
                f"-itsoffset 1.0 -i {video_path} "
                f"-map 0:v -map 1:a "
                f"-c copy {delayed_video} -loglevel error "
            )
        )

        # Get the duration of the video
        duration = subprocess.check_output(
            shlex.split(
                f"ffprobe -v error -show_entries "
                f"format=duration -of "
                f"default=noprint_wrappers=1:nokey=1 "
                f"{delayed_video}"
            )
        )
        duration = float(duration)

        # assert the length of a signal loaded from the
        # video and the audio are the same
        signal = AudioSignal.load_from_file_with_ffmpeg(delayed_video)
        assert np.allclose(signal.signal_duration, duration)
