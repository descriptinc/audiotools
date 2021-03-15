import torchaudio
import torch

class AudioSignal():
    def __init__(self, audio_path=None, audio_array=None, sample_rate=None, stft_params=None):
        pass


    # I/O
    def load_from_file(self, audio_path):
        data, sample_rate = torchaudio.load(audio_path)
        self.audio_data = data
        self.sample_rate = sample_rate
        return self

    def load_from_array(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        return self

    def write(self, audio_path):
        pass

    # 

    @property
    def audio_data(self):
        pass

    def stft(self):
        pass

    def istft(self):
        pass

    def transcribe(self):
        pass