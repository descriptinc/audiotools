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

import audiotools

css = Path(audiotools.__file__).parent / "core" / "templates" / "pandoc.css"


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


def audio_table(audio_dict):  # pragma: no cover
    FORMAT = "| Label | Audio \n" "|---|:-: \n"

    for k, v in audio_dict.items():
        formatted_audio = v.embed(display=False, return_html=True)
        row = f"| {k} | {formatted_audio} |\n"
        FORMAT += row

    return FORMAT


def discourse_audio_table(audio_dict, **kwargs):  # pragma: no cover
    """Creates a Markdown table out of a dictionary of
    AudioSignal objects which looks something like:

    | Label | Audio
    | [key1] | [val1.audio_data embedded]
    | [key2] | [val2.audio_data embedded]

    Parameters
    ----------
    audio_dict : dict[str, AudioSignal]
        Dictionary of strings mapped to AudioSignal objects.
    """
    FORMAT = "| Label | Audio \n" "|---|:-: \n"
    uploads = []

    for k, v in audio_dict.items():
        upload = v.upload_to_discourse(k, **kwargs)
        formatted_audio = upload[0].replace("|", "\|")
        row = f"| {k} | {formatted_audio} |\n"
        FORMAT += row
        uploads.append(upload)
    return FORMAT, uploads


def disp(obj, label=None):  # pragma: no cover
    from audiotools import AudioSignal

    DISCOURSE = bool(os.environ.get("UPLOAD_TO_DISCOURSE", False))

    if isinstance(obj, AudioSignal):
        if DISCOURSE:
            info = obj.upload_to_discourse(label=label, ext=".mp3")
            print(info[0])
        else:
            audio_elem = obj.embed(display=False, return_html=True)
            print(audio_elem)
    if isinstance(obj, dict):
        if DISCOURSE:
            table = discourse_audio_table(obj, ext=".mp3")[0]
        else:
            table = audio_table(obj)
        print(table)
    if isinstance(obj, plt.Figure):
        if DISCOURSE:
            info = upload_figure_to_discourse()
            print(info[0])
        else:
            plt.show()


@argbind.bind(without_prefix=True, positional=True)
def create_post(
    in_file: str,
    discourse: bool = False,
):  # pragma: no cover
    env = os.environ.copy()

    if not discourse:
        command = (
            f"codebraid pandoc --from markdown --to html "
            f"--css '{str(css)}' --standalone --wrap=none "
            f"--self-contained --no-cache"
        )
    else:
        env["UPLOAD_TO_DISCOURSE"] = str(int(discourse))
        command = (
            f"codebraid pandoc "
            f"--from markdown --to markdown "
            f"--wrap=none -t gfm --no-cache "
        )

    command += f" {in_file}"
    output = subprocess.check_output(shlex.split(command), env=env)
    print(output.decode(encoding="UTF-8"))


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        create_post()
