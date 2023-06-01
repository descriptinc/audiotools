---
file_format: mystnb
kernelspec:
  name: python3
---

# Creating listening tests

AudioTools comes up with a few utilities for creating
preference tests with best practices for audio playback. The
purpose of this tutorial is to teach you how to use them to create
your own listening tests on your data, and analyze the results.


## Folder structure

The utilities below assume the following folder structure:

```
folder
  condition_a/
    sample_0.wav
    sample_1.wav
    ...
  condition_b/
    sample_0.wav
    sample_1.wav
    ...
  some_other_name/
    sample_0.wav
    sample_1.wav
    ...
```

That is, audio files are kept organized inside the folder such that each subfolder corresponds to a different condition (e.g. output from a model). Samples names are kept consistent such that the same sample name corresponds to the same underlying audio across all conditions.

Let's make some dummy data that follows the rules here, and also import everything we need to make tests:

```{code-cell} ipython3
import math
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import rich

from audiotools import preference as pr

@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: list = None
    reference: str = None
    seed: int = 0

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

config = Config(
    folder="/tmp/pref/audio/",
    save_path="/tmp/pref/results.csv",
    conditions=["condition_a", "condition_b"],
    reference="condition_c",
)

create_data(config.folder)
```

## Loading the data

Now that we have some data in a folder, we'll
use the `Samples` object in `audiotools.preference` to find all the audio, and
organize it by condition.

```{code-cell} ipython3
from audiotools import preference as pr

data = pr.Samples(config.folder)
```

Inside `data` is a dictionary containing all the audio samples, organized
by condition, as well as a list of sample names which the test will iterate through. The samples are shuffled by default.

```{code-cell} ipython3
rich.print(data.samples)
rich.print(data.names)
```

The `Samples` object also contains information about the state of the test, such
as the current sample, and has utilities for getting the next sample, etc.

## Adding the Player

The Player object lets you easily create an audio player with play buttons for every audio file, region selection, and looping. The player audio requires
a Gradio `app` object to be passed to it on initialization.

```{code-cell} ipython3
with gr.Blocks() as app:
    player = pr.Player(app)
```

This instantiates the player, and adds relevant Javascript and CSS to the Gradio
app. Next, let's actually create the player, and add buttons for playback:

```{code-cell} ipython3
with gr.Blocks() as app:
    player = pr.Player(app)
    player.create()
    player.add("Play Reference")
```

This created a player, added it to the app, and then added a button called "Play Reference", with an underlying (invisible) audio element. To actually set the audio for the element, we can do this:

```{code-cell} ipython3
with gr.Blocks() as app:
    player = pr.Player(app)
    player.create()
    player.add("Play Reference")
    player.to_list()[0].update(value=data.samples["sample_0.wav"]["condition_a"])
```

Now, when you hit play, it'll play back that audio file. The order of audio files is always the order in which they were added to the player.

## Tracking users

We can track users of the app by using the `create_tracker` function. This function creates a hidden text box with a user id in it that is saved as a
cookie in the user's browser. Enable it like this:

```{code-cell} ipython3
with gr.Blocks() as app:
    user = pr.create_tracker(app)
```

Then, you can just use `user` as an input to any Gradio function, and it will contain the value of the cookie.

## ABX Preference script

Let's put everything together to create a simple ABX-based preference test.
Copy paste the code below and launch it to see the preference test.

```{code-cell} ipython3
import math
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import rich

from audiotools import preference as pr

@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: list = None
    reference: str = None
    seed: int = 0

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

config = Config(
    folder="/tmp/pref/audio/",
    save_path="/tmp/pref/results.csv",
    conditions=["condition_a", "condition_b"],
    reference="condition_c",
)

create_data(config.folder)

with gr.Blocks() as app:
    save_path = config.save_path
    samples = gr.State(pr.Samples(config.folder))

    reference = config.reference
    conditions = config.conditions
    assert len(conditions) == 2, "Preference tests take only two conditions!"

    player = pr.Player(app)
    player.create()
    if reference is not None:
        player.add("Play Reference")

    user = pr.create_tracker(app)

    with gr.Row().style(equal_height=True):
        for i in range(len(conditions)):
            x = string.ascii_uppercase[i]
            player.add(f"Play {x}")

    rating = gr.Slider(value=50, interactive=True)
    gr.HTML(pr.slider_abx)

    def build(user, samples, rating):
        samples.filter_completed(user, save_path)

        # Write results to CSV
        if samples.current > 0:
            start_idx = 1 if reference is not None else 0
            name = samples.names[samples.current - 1]
            result = {"sample": name, "user": user}

            result[samples.order[start_idx]] = 100 - rating
            result[samples.order[start_idx + 1]] = rating
            pr.save_result(result, save_path)

        updates, done, pbar = samples.get_next_sample(reference, conditions)
        return updates + [gr.update(value=50), done, samples, pbar]

    progress = gr.HTML()
    begin = gr.Button("Submit", elem_id="start-survey")
    begin.click(
        fn=build,
        inputs=[user, samples, rating],
        outputs=player.to_list() + [rating, begin, samples, progress],
    ).then(None, _js=pr.reset_player)

    # Comment this back in to actually launch the script.
    # app.launch()
```

## MUSHRA listening test

```{code-cell} ipython3
import math
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import rich

from audiotools import preference as pr

@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: list = None
    reference: str = None
    seed: int = 0

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

config = Config(
    folder="/tmp/pref/audio/",
    save_path="/tmp/pref/results.csv",
    conditions=["condition_a", "condition_b"],
    reference="condition_c",
)

create_data(config.folder)

with gr.Blocks() as app:
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
        if samples.current > 0:
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

    # Comment this back in to actually launch the script.
    # app.launch()
```

Feel free to mix and match stuff from the scripts above to create one suited for your own needs!
