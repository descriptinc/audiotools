import tempfile
import typing
import zipfile
from pathlib import Path

import markdown2 as md
import matplotlib.pyplot as plt
import torch
from IPython.display import HTML


def audio_zip(audio_dict: dict, zip_path: str, **kwargs):
    """Creates a zip file based on a dictionary of audio signals
    with both waveforms and spectrogram images in it. The dictionary
    can be constructed (for example) like this:

    >>> audio_dict = defaultdict(lambda: [])
    >>> audio_signals = ... # some list of audio signals
    >>> models = ... # some list of models
    >>> for signal in audio_signals:
    >>>     audio_dict["input"].append(signal.clone())
    >>>     for i, model in enumerate(models):
    >>>         output = model(signal)
    >>>         audio_dict[f"model_{i}"].append(output.clone())
    >>> audiotools.post.audio_zip(audio_dict, "samples.zip")

    Then, the zip file can be easily shared.

    Parameters
    ----------
    audio_dict : dict
        Dictionary containing keys which will be folders in the zip file,
        and lists of AudioSignals which will be written to the folders
        in the zip file.
    zip_path : str
        Path to place the zip file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for k, signals in audio_dict.items():
                for i, signal in enumerate(signals):
                    with open(tmpdir / "out.wav", "w") as f:
                        signal.write(f.name)
                        zip_file.write(f.name, Path(k) / f"sample_{i}.wav")
                    with open(tmpdir / "out.png", "w") as f:
                        signal.save_image(f.name, **kwargs)
                        zip_file.write(f.name, Path(k) / f"sample_{i}.png")


def audio_table(
    audio_dict: dict,
    first_column: str = None,
    format_fn: typing.Callable = None,
    **kwargs,
):  # pragma: no cover
    """Embeds an audio table into HTML, or as the output cell
    in a notebook.

    Parameters
    ----------
    audio_dict : dict
        Dictionary of data to embed.
    first_column : str, optional
        The label for the first column of the table, by default None
    format_fn : typing.Callable, optional
        How to format the data, by default None

    Returns
    -------
    str
        Table as a string

    Examples
    --------

    >>> audio_dict = {}
    >>> for i in range(signal_batch.batch_size):
    >>>     audio_dict[i] = {
    >>>         "input": signal_batch[i],
    >>>         "output": output_batch[i]
    >>>     }
    >>> audiotools.post.audio_table(audio_dict)

    """
    from audiotools import AudioSignal

    output = []
    columns = None

    def _default_format_fn(label, x, **kwargs):
        if torch.is_tensor(x):
            x = x.tolist()

        if x is None:
            return "."
        elif isinstance(x, AudioSignal):
            return x.embed(display=False, return_html=True, **kwargs)
        else:
            return str(x)

    if format_fn is None:
        format_fn = _default_format_fn

    if first_column is None:
        first_column = "."

    for k, v in audio_dict.items():
        if not isinstance(v, dict):
            v = {"Audio": v}

        v_keys = list(v.keys())
        if columns is None:

            columns = [first_column] + v_keys
            output.append(" | ".join(columns))

            layout = "|---" + len(v_keys) * "|:-:"
            output.append(layout)

        formatted_audio = []
        for col in columns[1:]:
            formatted_audio.append(format_fn(col, v[col], **kwargs))

        row = f"| {k} | "
        row += " | ".join(formatted_audio)
        output.append(row)

    output = "\n" + "\n".join(output)
    return output


def in_notebook():  # pragma: no cover
    """Determines if code is running in a notebook.

    Returns
    -------
    bool
        Whether or not this is running in a notebook.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def disp(obj, **kwargs):  # pragma: no cover
    """Displays an object, depending on if its in a notebook
    or not.

    Parameters
    ----------
    obj : typing.Any
        Any object to display.

    """
    from audiotools import AudioSignal

    IN_NOTEBOOK = in_notebook()

    if isinstance(obj, AudioSignal):
        audio_elem = obj.embed(display=False, return_html=True)
        if IN_NOTEBOOK:
            return HTML(audio_elem)
        else:
            print(audio_elem)
    if isinstance(obj, dict):
        table = audio_table(obj, **kwargs)
        if IN_NOTEBOOK:
            return HTML(md.markdown(table, extras=["tables"]))
        else:
            print(table)
    if isinstance(obj, plt.Figure):
        plt.show()
