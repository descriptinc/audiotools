import torchaudio
torchaudio.set_audio_backend("soundfile")

from .audio_signal import AudioSignal, STFTParams