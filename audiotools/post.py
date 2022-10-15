import tempfile
import zipfile
from pathlib import Path

import markdown2 as md
import matplotlib.pyplot as plt
import torch
from IPython.display import HTML


def audio_zip(audio_dict, zip_path, **kwargs):
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
    audio_dict, first_column=None, format_fn=None, **kwargs
):  # pragma: no cover
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
