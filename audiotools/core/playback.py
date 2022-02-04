"""
These are optional utilities included in nussl that allow one to embed an AudioSignal
as a playable object in a Jupyter notebook, or to play audio from
the terminal.
"""
import base64
import io
import random
import string
import subprocess
from copy import deepcopy
from tempfile import NamedTemporaryFile

import importlib_resources as pkg_resources
import matplotlib.pyplot as plt
from PIL import Image

from . import templates
from .util import _close_temp_files

headers = pkg_resources.read_text(templates, "headers.html")
widget = pkg_resources.read_text(templates, "widget.html")


def _check_imports():  # pragma: no cover
    try:
        import ffmpy
    except:
        ffmpy = False

    try:
        import IPython
    except:
        raise ImportError("IPython must be installed in order to use this function!")
    return ffmpy, IPython


class PlayMixin:
    def embed(self, batch_idx=0, ext=".mp3", display=True):
        """
        Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3
        into the notebook.

        Args:
            audio_signal (AudioSignal): AudioSignal object containing the data.
            ext (str): What extension to use when embedding. '.mp3' is more lightweight
            leading to smaller notebook sizes. Defaults to '.mp3'.
            display (bool): Whether or not to display the object immediately, or to return
            the html object for display later by the end user. Defaults to True.

        Example:
            >>> import nussl
            >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
            >>> audio_signal = nussl.AudioSignal(audio_file)
            >>> audio_signal.embed_audio()

        This will show a little audio player where you can play the audio inline in
        the notebook.
        """
        ext = f".{ext}" if not ext.startswith(".") else ext
        ffmpy, IPython = _check_imports()
        sr = self.sample_rate
        tmpfiles = []

        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(mode="w+", suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name, batch_idx=batch_idx)
            if ext != ".wav" and ffmpy:
                tmp_converted = NamedTemporaryFile(mode="w+", suffix=ext, delete=False)
                tmpfiles.append(tmp_wav)
                ff = ffmpy.FFmpeg(
                    inputs={tmp_wav.name: None},
                    outputs={
                        tmp_converted.name: "-write_xing 0 -codec:a libmp3lame -b:a 128k -y -hide_banner -loglevel error"
                    },
                )
                ff.run()
            else:
                tmp_converted = tmp_wav

            audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
            if display:
                IPython.display.display(audio_element)
        return audio_element

    def widget(
        self,
        title=None,
        batch_idx=0,
        ext=".mp3",
        add_headers=True,
        player_width="100%",
        max_width="600px",
        margin="10px",
        plot_fn=None,
        fig_size=(12, 4),
        return_html=False,
        **kwargs,
    ):
        """Creates a playable widget with spectrogram. Inspired (heavily) by
        https://sjvasquez.github.io/blog/melnet/.

        Parameters
        ----------
        batch_idx : int, optional
            Which item in batch to display, by default 0
        ext : str, optional
            Extension for embedding, by default ".mp3"
        display : bool, optional
            Whether or not to display the widget, by default True
        add_headers : bool, optional
            Whether or not to add headers (use for first embed, False for later embeds), by default True
        player_width : str, optional
            Width of the player, as a string in a CSS rule, by default "100%"
        max_width : str, optional
            Maximum width of player, by default "600px"
        margin : str, optional
            Margin on all sides of player, by default "10px"
        plot_fn : function, optional
            Plotting function to use (by default self.specshow).
        title : str, optional
            Title of plot, placed in upper right of top-most axis.
        fig_size : tuple, optional
            Size of figure.
        kwargs : dict, optional
            Keyword arguments to plot_fn (by default self.specshow).

        Returns
        -------
        HTML
            HTML object.
        """

        def _adjust_figure(fig):
            fig.set_size_inches(*fig_size)
            plt.ioff()

            axs = fig.axes
            for ax in axs:
                ax.margins(0, 0)
                ax.set_axis_off()
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        def _save_fig_to_tag():
            buffer = io.BytesIO()

            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close()

            buffer.seek(0)
            data_uri = base64.b64encode(buffer.read()).decode("ascii")
            tag = "data:image/png;base64,{0}".format(data_uri)

            return tag

        _, IPython = _check_imports()

        header_html = ""

        if add_headers:
            header_html = headers.replace("PLAYER_WIDTH", str(player_width))
            header_html = header_html.replace("MAX_WIDTH", str(max_width))
            header_html = header_html.replace("MARGIN", str(margin))
            IPython.display.display(IPython.display.HTML(header_html))

        widget_html = widget

        if plot_fn is None:
            plot_fn = self.specshow
            kwargs["batch_idx"] = batch_idx
        plot_fn(**kwargs)

        fig = plt.gcf()
        axs = fig.axes
        _adjust_figure(fig)

        if title is not None:
            t = axs[0].annotate(
                title,
                xy=(1, 1),
                xycoords="axes fraction",
                fontsize=25,
                xytext=(-5, -5),
                textcoords="offset points",
                ha="right",
                va="top",
                color="white",
            )
            t.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="black"))

        tag = _save_fig_to_tag()

        # Make the source image for the levels
        fig = plt.figure()
        self.specshow(batch_idx=batch_idx)
        _adjust_figure(fig)
        levels_tag = _save_fig_to_tag()

        player_id = "".join(random.choice(string.ascii_uppercase) for _ in range(10))

        audio_elem = self.embed(batch_idx=batch_idx, ext=ext, display=False)
        widget_html = widget_html.replace("AUDIO_SRC", audio_elem.src_attr())
        widget_html = widget_html.replace("IMAGE_SRC", tag)
        widget_html = widget_html.replace("LEVELS_SRC", levels_tag)
        widget_html = widget_html.replace("PLAYER_ID", player_id)

        # Calculate height of figure based on figure size.
        padding_amount = str(fig_size[1] * 9) + "%"
        widget_html = widget_html.replace("PADDING_AMOUNT", padding_amount)

        IPython.display.display(IPython.display.HTML(widget_html))

        if return_html:
            html = header_html if add_headers else ""
            html += widget_html
            return html

    def play(self, batch_idx=0):
        """
        Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
        Otherwise, will fail. The audio signal is written to a temporary file
        and then played with ffplay.

        Args:
            audio_signal (AudioSignal): AudioSignal object to be played.
        """
        tmpfiles = []
        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name, batch_idx=batch_idx)
            print(self)
            subprocess.call(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    tmp_wav.name,
                ]
            )
        return self


if __name__ == "__main__":
    from audiotools import AudioSignal
    from matplotlib.gridspec import GridSpec

    signal = AudioSignal(
        "tests/audio/spk/f10_script4_produced.mp3", offset=5, duration=5
    )

    wave_html = signal.widget(
        "Waveform plot", plot_fn=signal.waveplot, return_html=True, fig_size=(12, 2)
    )

    spec_html = signal.widget("Spectrogram plot", return_html=True, add_headers=False)

    def plot_fn():
        gs = GridSpec(6, 1)
        plt.subplot(gs[0, :])
        signal.waveplot()
        plt.subplot(gs[1:, :])
        signal.specshow()

    combined_html = signal.widget(
        "Combined plot",
        plot_fn=plot_fn,
        return_html=True,
        fig_size=(12, 5),
        add_headers=False,
    )

    with open("/tmp/index.html", "w") as f:
        f.write(wave_html)
        f.write(spec_html)
        f.write(combined_html)
