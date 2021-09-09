import tempfile
import timeit

import librosa
import torch
import torchaudio

from audiotools import AudioSignal
from audiotools import util


class LibrosaSignal(AudioSignal):
    def load_from_file(self, audio_path, offset, duration, device=None):
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


class TorchSignal(AudioSignal):
    def load_from_file(self, audio_path, offset, duration, device=None):
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

        self.audio_data = data
        self.original_signal_length = self.signal_length

        self.audio_mask = torch.ones_like(self.audio_data)
        self.sample_rate = sample_rate
        self.path_to_input_file = audio_path
        return self.to(device)


# Load 2 second excerpt from a 2 hour file
with tempfile.NamedTemporaryFile(suffix=".wav") as f:
    signal = AudioSignal(torch.randn(44100 * 60 * 2), 44100)
    signal.write(f.name)

    def func():
        LibrosaSignal.excerpt(f.name, duration=2.0)

    librosa_time = timeit.timeit(func, number=10)

    def func():
        TorchSignal.excerpt(f.name, duration=2.0)

    torch_time = timeit.timeit(func, number=10)

    print(f"Librosa loading took {librosa_time}")
    print(f"Torch loading took {torch_time}")

    print(f"Librosa is {torch_time / librosa_time}x faster than Torch")
