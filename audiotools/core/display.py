import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .. import post


class DisplayMixin:
    def specshow(self, batch_idx=0, x_axis="time", y_axis="linear", **kwargs):
        import librosa
        import librosa.display

        # Always re-compute the STFT data before showing it, in case
        # it changed.
        self.stft_data = None

        log_mag = librosa.amplitude_to_db(self.magnitude.cpu().numpy(), ref=np.max)
        librosa.display.specshow(
            log_mag[batch_idx].mean(axis=0),
            x_axis=x_axis,
            y_axis=y_axis,
            sr=self.sample_rate,
            **kwargs,
        )

    def waveplot(self, batch_idx=0, x_axis="time", **kwargs):
        import librosa
        import librosa.display

        audio_data = self.audio_data[batch_idx].mean(dim=0)
        audio_data = audio_data.cpu().numpy()

        plot_fn = "waveshow" if hasattr(librosa.display, "waveshow") else "waveplot"
        wave_plot_fn = getattr(librosa.display, plot_fn)
        wave_plot_fn(audio_data, x_axis=x_axis, sr=self.sample_rate, **kwargs)

    def wavespec(self, batch_idx=0, x_axis="time", **kwargs):
        gs = GridSpec(6, 1)
        plt.subplot(gs[0, :])
        self.waveplot(batch_idx=batch_idx, x_axis=x_axis)
        plt.subplot(gs[1:, :])
        self.specshow(batch_idx=batch_idx, x_axis=x_axis, **kwargs)

    def upload_to_discourse(
        self,
        label=None,
        api_username=None,
        api_key=None,
        batch_idx=0,
        discourse_server=None,
        ext=".wav",
    ):  # pragma: no cover
        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            self.write(f.name, batch_idx=batch_idx)

            info = post.upload_file_to_discourse(
                f.name,
                api_username=api_username,
                api_key=api_key,
                discourse_server=discourse_server,
            )

            label = self.path_to_input_file if label is None else label
            if label is None:
                label = "unknown"

            formatted = f"![{label}|audio]({info['short_path']})"
            return formatted, info
