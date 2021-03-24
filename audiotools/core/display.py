import librosa
import librosa.display
import numpy as np

class DisplayMixin:
    def specshow(self, batch_idx=0, x_axis='time', y_axis='linear', **kwargs):
        log_mag = librosa.amplitude_to_db(
            self.magnitude.cpu().numpy(), ref=np.max
        )
        librosa.display.specshow(
            log_mag[batch_idx].mean(axis=0), 
            x_axis=x_axis, 
            y_axis=y_axis,
            **kwargs
        )

    def waveplot(self, batch_idx=0, x_axis='time', **kwargs):
        audio_data = self.audio_data[batch_idx].mean(dim=0)
        audio_data = audio_data.cpu().numpy()
        librosa.display.waveplot(
            audio_data, x_axis=x_axis, **kwargs
        )
