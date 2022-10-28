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
        audio_path = Path(d) / "test.wav"
        delayed_video = Path(d) / "test_delayed.mp4"
        delayed_audio = Path(d) / "test_audio.wav"
        remuxed_video = Path(d) / "test_remuxed.mp4"

        # Create a test video
        subprocess.run(
            shlex.split(
                f"ffmpeg -y -f lavfi "
                f"-i testsrc=d=5:s=120x120:r=60,format=yuv420p "
                f"-f lavfi -i sine=f=440:b=4 "
                f"-shortest {video_path} -loglevel error"
            )
        )

        signal = AudioSignal(video_path)
        signal.write(audio_path)

        # Create a video with the audio offset by 1 second
        subprocess.run(
            shlex.split(
                f"ffmpeg -y -i {video_path} "
                f"-itsoffset 1.0 -i {audio_path} "
                f"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "
                f"{delayed_video} -loglevel error "
            )
        )
        signal = AudioSignal.load_from_file_with_ffmpeg(delayed_video)

        # Mux the read signal with the video, and then re-read it
        # to make sure it stays the same.
        signal.write(delayed_audio)
        subprocess.run(
            shlex.split(
                f"ffmpeg -i {delayed_video} "
                f"-i {delayed_audio} -c:v "
                f"copy -c:a aac -map 0:v:0 "
                f"-map 1:a:0 {remuxed_video} -loglevel error"
            )
        )
        remuxed = AudioSignal.load_from_file_with_ffmpeg(remuxed_video)

        # Muxing encodes the audio, changing it so the best
        # we can do is compare the first nonzero offset (which
        # is the encoded delay)
        idx_a = signal.audio_data[0, 0].nonzero()[0]
        idx_b = remuxed.audio_data[0, 0].nonzero()[0]
        # Error of less than 50 samples
        assert abs(idx_a - idx_b) < 50
