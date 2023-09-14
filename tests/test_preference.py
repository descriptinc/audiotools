import math
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from audiotools import preference as pr


@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: list = None
    reference: str = None
    seed: int = 0


#######################################################
### Create audio data for debugging listening tests ###
#######################################################


def random_sine(f):
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 5.0  # in seconds, may be float

    # generate samples, note conversion to float32 array
    volume = 0.1
    num_samples = int(fs * duration)
    samples = volume * np.sin(2 * math.pi * (f / fs) * np.arange(num_samples))

    return samples, fs


def create_data(path):
    path = Path(path)
    hz = [110, 140, 180]

    for i in range(6):
        name = f"condition_{string.ascii_lowercase[i]}"
        for j in range(3):
            sample_path = path / name / f"sample_{j}.wav"
            sample_path.parent.mkdir(exist_ok=True, parents=True)
            audio, sr = random_sine(hz[j] * (2**i))
            sf.write(sample_path, audio, sr)


def _test_mushra(app, config):
    """Launches a MUSHRA test, with an optional reference."""
    save_path = config.save_path
    samples = gr.State(pr.Samples(config.folder))

    reference = config.reference
    conditions = config.conditions

    player = pr.Player(app)
    player.create()
    if reference is not None:
        player.add("Play Reference")

    user = pr.create_tracker(app)
    ratings = []

    with gr.Row():
        gr.HTML("")
        with gr.Column(scale=9):
            gr.HTML(pr.slider_mushra)

    for i in range(len(conditions)):
        with gr.Row().style(equal_height=True):
            x = string.ascii_uppercase[i]
            player.add(f"Play {x}")
            with gr.Column(scale=9):
                ratings.append(gr.Slider(value=50, interactive=True))

    def build(user, samples, *ratings):
        # Filter out samples user has done already, by looking in the CSV.
        samples.filter_completed(user, save_path)

        # Write results to CSV
        if samples.current > 0 and len(samples.names) > 0:
            start_idx = 1 if reference is not None else 0
            name = samples.names[samples.current - 1]
            result = {"sample": name, "user": user}
            for k, r in zip(samples.order[start_idx:], ratings):
                result[k] = r
            pr.save_result(result, save_path)

        updates, done, pbar = samples.get_next_sample(reference, conditions)
        return updates + [gr.update(value=50) for _ in ratings] + [done, samples, pbar]

    progress = gr.HTML()
    begin = gr.Button("Submit", elem_id="start-survey")
    begin.click(
        fn=build,
        inputs=[user, samples] + ratings,
        outputs=player.to_list() + ratings + [begin, samples, progress],
    ).then(None, _js=pr.reset_player)

    samples = pr.Samples(config.folder)
    for i in range(len(samples) + 1):
        build("test", samples, 95, 85)


def test_preference():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config = Config(
            folder=tmpdir,
            save_path=tmpdir / "results.csv",
            conditions=["condition_a", "condition_b"],
            reference="condition_c",
        )

        create_data(config.folder)
        with gr.Blocks() as app:
            _test_mushra(app, config)
            _test_mushra(app, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config = Config(
            folder=tmpdir,
            save_path=tmpdir / "results.csv",
            conditions=["condition_a", "condition_b"],
        )

        create_data(config.folder)
        with gr.Blocks() as app:
            _test_mushra(app, config)
