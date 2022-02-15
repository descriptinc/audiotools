"""Helper script for creating audio-enriched HTML/Discourse posts.
"""
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import argbind
import matplotlib.pyplot as plt


def upload_file_to_discourse(
    path, api_username=None, api_key=None, discourse_server=None
):  # pragma: no cover
    if api_username is None:
        api_username = os.environ.get("DISCOURSE_API_USERNAME", None)
    if api_key is None:
        api_key = os.environ.get("DISCOURSE_API_KEY", None)
    if discourse_server is None:
        discourse_server = os.environ.get("DISCOURSE_SERVER", None)

    if discourse_server is None or api_key is None or api_username is None:
        raise RuntimeError(
            "DISCOURSE_API_KEY, DISCOURSE_SERVER, DISCOURSE_API_USERNAME must be set in your environment!"
        )

    command = (
        f"curl -s -X POST {discourse_server}/uploads.json "
        f"-H 'content-type: multipart/form-data;' "
        f"-H 'Api-Key: {api_key}' "
        f"-H 'Api-Username: {api_username}' "
        f"-F 'type=composer' "
        f"-F 'files[]=@{path}' "
    )
    return json.loads(subprocess.check_output(shlex.split(command)))


class DiscourseMixin:
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

            info = upload_file_to_discourse(
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


def upload_figure_to_discourse(
    label=None,
    fig=None,
    bbox_inches="tight",
    pad_inches=0,
    api_username=None,
    api_key=None,
    discourse_server=None,
    **kwargs,
):  # pragma: no cover
    if fig is None:
        fig = plt.gcf()

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        plt.savefig(f.name, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

        info = upload_file_to_discourse(
            f.name,
            api_username=api_username,
            api_key=api_key,
            discourse_server=discourse_server,
        )

    if label is None:
        label = "unknown"
        formatted = f"![{label}|image]({info['short_path']})"
    return formatted, info


def audio_table(audio_dict, first_column=None, format_fn=None):  # pragma: no cover
    output = []
    columns = None

    def _default_format_fn(label, x):
        if x is None:
            return "."
        return x.embed(display=False, return_html=True)

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
            formatted_audio.append(format_fn(col, v[col]))

        row = f"| {k} | "
        row += " | ".join(formatted_audio)
        output.append(row)

    output = "\n".join(output)
    return output


def discourse_audio_table(audio_dict, first_column=None, **kwargs):  # pragma: no cover
    """Creates a Markdown table out of a dictionary of
    AudioSignal objects.

    Parameters
    ----------
    audio_dict : dict[str, dict]
        Dictionary of strings mapped to dictionaries of AudioSignal objects.
    """
    uploads = []

    def format_fn(label, x):
        if x is None:
            return "."
        upload = x.upload_to_discourse(label, **kwargs)
        formatted_audio = upload[0].replace("|", "\|")
        uploads.append(upload)
        return formatted_audio

    output = audio_table(audio_dict, first_column=first_column, format_fn=format_fn)
    return output, uploads


def disp(obj, label=None, **kwargs):  # pragma: no cover
    from audiotools import AudioSignal

    DISCOURSE = bool(os.environ.get("UPLOAD_TO_DISCOURSE", False))

    if isinstance(obj, AudioSignal):
        if DISCOURSE:
            info = obj.upload_to_discourse(label=label, **kwargs)
            print(info[0])
        else:
            audio_elem = obj.embed(display=False, return_html=True)
            print(audio_elem)
    if isinstance(obj, dict):
        if DISCOURSE:
            table = discourse_audio_table(obj, **kwargs)[0]
        else:
            table = audio_table(obj)
        print(table)
    if isinstance(obj, plt.Figure):
        if DISCOURSE:
            info = upload_figure_to_discourse()
            print(info[0])
        else:
            plt.show()


def create_post(
    in_file: str,
    discourse: bool = False,
    use_cache: bool = False,
):  # pragma: no cover
    env = os.environ.copy()

    import audiotools

    css = Path(audiotools.__file__).parent / "core" / "templates" / "pandoc.css"

    if not discourse:
        command = (
            f"codebraid pandoc --from markdown --to html "
            f"--css '{str(css)}' --standalone --wrap=none "
            f"--self-contained "
        )
    else:
        env["UPLOAD_TO_DISCOURSE"] = str(int(discourse))
        command = (
            f"codebraid pandoc " f"--from markdown --to markdown " f"--wrap=none -t gfm"
        )

    if not use_cache:
        command += " --no-cache"

    command += f" {in_file}"
    output = subprocess.check_output(shlex.split(command), env=env)
    print(output.decode(encoding="UTF-8"))


if __name__ == "__main__":  # pragma: no cover
    create_post = argbind.bind(create_post, without_prefix=True, positional=True)
    args = argbind.parse_args()
    with argbind.scope(args):
        create_post()
