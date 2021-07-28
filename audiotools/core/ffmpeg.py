import shlex
import subprocess
import tempfile

import numpy as np
import torch


def r128stats(filepath, quiet):
    """takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter. Taken lovingly from Scaper."""
    ffargs = [
        "ffmpeg",
        "-nostats",
        "-i",
        filepath,
        "-filter_complex",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    if quiet:
        ffargs += ["-hide_banner"]
    proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE, universal_newlines=True)
    stats = proc.communicate()[1]
    summary_index = stats.rfind("Summary:")

    summary_list = stats[summary_index:].split()
    i_lufs = float(summary_list[summary_list.index("I:") + 1])
    i_thresh = float(summary_list[summary_list.index("I:") + 4])
    lra = float(summary_list[summary_list.index("LRA:") + 1])
    lra_thresh = float(summary_list[summary_list.index("LRA:") + 4])
    lra_low = float(summary_list[summary_list.index("low:") + 1])
    lra_high = float(summary_list[summary_list.index("high:") + 1])
    stats_dict = {
        "I": i_lufs,
        "I Threshold": i_thresh,
        "LRA": lra,
        "LRA Threshold": lra_thresh,
        "LRA Low": lra_low,
        "LRA High": lra_high,
    }

    return stats_dict


class FFMPEGMixin:
    _loudness = None

    def ffmpeg_loudness(self, quiet=True):
        loudness = []

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            for i in range(self.batch_size):
                self.write(f.name, i)
                loudness_stats = r128stats(f.name, quiet=quiet)
                loudness.append(loudness_stats["I"])

        self._loudness = torch.from_numpy(np.array(loudness))
        return self.loudness()

    def ffmpeg_resample(self, sample_rate, quiet=True):
        from audiotools import AudioSignal

        if sample_rate == self.sample_rate:
            return self

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name)
            f_out = f.name.replace("wav", "rs.wav")
            command = f"ffmpeg -i {f.name} -ar {sample_rate} {f_out}"
            if quiet:
                command += " -hide_banner -loglevel error"
            subprocess.check_call(shlex.split(command))
            resampled = AudioSignal(f_out)
        return resampled
