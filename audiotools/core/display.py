import inspect
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from . import util


def format_figure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        f_keys = inspect.signature(util.format_figure).parameters.keys()
        f_kwargs = {}
        for k, v in list(kwargs.items()):
            if k in f_keys:
                kwargs.pop(k)
                f_kwargs[k] = v
        func(*args, **kwargs)
        util.format_figure(**f_kwargs)

    return wrapper


class DisplayMixin:
    @format_figure
    def specshow(
        self,
        no_format=False,
        title=None,
        batch_idx=0,
        preemphasis=True,
        x_axis="time",
        y_axis="linear",
        **kwargs,
    ):
        import librosa
        import librosa.display

        # Always re-compute the STFT data before showing it, in case
        # it changed.
        signal = self.clone()
        signal.stft_data = None
        if preemphasis:
            signal.preemphasis()

        log_mag = librosa.amplitude_to_db(signal.magnitude.cpu().numpy(), ref=np.max)
        librosa.display.specshow(
            log_mag[batch_idx].mean(axis=0),
            x_axis=x_axis,
            y_axis=y_axis,
            sr=signal.sample_rate,
            **kwargs,
        )

    @format_figure
    def waveplot(self, title=None, batch_idx=0, x_axis="time", **kwargs):
        import librosa
        import librosa.display

        audio_data = self.audio_data[batch_idx].mean(dim=0)
        audio_data = audio_data.cpu().numpy()

        plot_fn = "waveshow" if hasattr(librosa.display, "waveshow") else "waveplot"
        wave_plot_fn = getattr(librosa.display, plot_fn)
        wave_plot_fn(audio_data, x_axis=x_axis, sr=self.sample_rate, **kwargs)

    @format_figure
    def wavespec(self, batch_idx=0, x_axis="time", **kwargs):
        gs = GridSpec(6, 1)
        plt.subplot(gs[0, :])
        self.waveplot(batch_idx=batch_idx, x_axis=x_axis)
        plt.subplot(gs[1:, :])
        self.specshow(batch_idx=batch_idx, x_axis=x_axis, **kwargs)

    def write_audio_to_tb(
        self,
        tag,
        writer,
        step: int = None,
        batch_idx: int = 0,
        plot_fn="specshow",
        **kwargs,
    ):
        audio_data = self.audio_data[batch_idx, 0].detach().cpu()
        sample_rate = self.sample_rate
        writer.add_audio(tag, audio_data, step, sample_rate)

        if plot_fn is not None:
            if isinstance(plot_fn, str):
                plot_fn = getattr(self, plot_fn)
                kwargs["batch_idx"] = batch_idx
            fig = plt.figure()
            plt.clf()
            plot_fn(**kwargs)
            writer.add_figure(tag.replace("wav", "png"), fig, step)

    def save_image(self, image_path: str, plot_fn="specshow", **kwargs):
        if isinstance(plot_fn, str):
            plot_fn = getattr(self, plot_fn)

        plt.clf()
        plot_fn(**kwargs)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()
