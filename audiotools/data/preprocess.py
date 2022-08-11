import csv
import os
from pathlib import Path

from ..core import AudioSignal


def create_csv(
    audio_files: list, output_csv: Path, loudness: bool = False, data_path: str = None
):
    """Converts a folder of audio files to a CSV file.

    Parameters
    ----------
    audio_files : list
        List of audio files.
    output_csv : Path
        Output CSV, with each row containing the relative path of every file
        to PATH_TO_DATA (defaults to None).
    loudness : bool
        Compute loudness of entire file and store alongside path.
    """
    if data_path is None:
        data_path = Path(os.getenv("PATH_TO_DATA", ""))

    info = []
    for af in audio_files:
        af = Path(af)
        _info = {}
        _info["path"] = af.relative_to(data_path)
        if loudness:
            _info["loudness"] = AudioSignal(af).ffmpeg_loudness().item()

        info.append(_info)

    with open(output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(info[0].keys()))
        writer.writeheader()

        for item in info:
            writer.writerow(item)
