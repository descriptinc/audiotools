---
file_format: mystnb
kernelspec:
  name: python3
---

# Transforms

This notebook explains the AudioTools transforms, how they work, how
they can be combined, and how to implement your own. It also shows a
full complete working example.

```{code-cell} ipython3
import audiotools
from audiotools import AudioSignal
from audiotools import post, util, metrics
from audiotools.data import preprocess
from flatten_dict import flatten
import torch
import pprint
from collections import defaultdict
from audiotools import transforms as tfm
import os

audiotools.core.playback.DEFAULT_EXTENSION = ".mp3"
util.DEFAULT_FIG_SIZE = (9, 3)
os.environ["PATH_TO_DATA"] = os.path.abspath("../..")

pp = pprint.PrettyPrinter()

def make_dict(signal_batch, output_batch, kwargs=None):
    audio_dict = {}

    kwargs_ = {}

    if kwargs is not None:
        kwargs = flatten(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                key = ".".join(list(k[-2:]))
                kwargs_[key] = v

    for i in range(signal_batch.batch_size):
        audio_dict[i] = {
            "input": signal_batch[i],
            "output": output_batch[i]
        }
        for k, v in kwargs_.items():
            try:
                audio_dict[i][k] = v[i].item()
            except:
                audio_dict[i][k] = v[i].float().mean()

    return audio_dict
```

## Quick start

Transforms are one of the biggest features in AudioTools, allowing for high-quality,
fast, GPU-powered audio augmentations, that can create very realistic simulated conditions.
Let's take an AudioSignal, and apply a sequence of transforms to it:

```{code-cell} ipython3
signal = AudioSignal("../../tests/audio/spk/f10_script4_produced.wav", offset=10, duration=5)
t = tfm.Compose(
    tfm.LowPass(),
    tfm.ClippingDistortion(),
    tfm.TimeMask(),
)
kwargs = t.instantiate(state=0, signal=signal) # Instantiate random parameters
output = t(signal.clone(), **kwargs) # Apply transform

signal.widget("Original")
output.widget("Transformed")
```

## Audio examples of all transforms

Below is a table demonstrating every transform we currently have implemented, with
randomly chosen parameters.

```{code-cell} ipython3
seed = 0

transforms_to_demo = []
for x in dir(tfm):
    if hasattr(getattr(tfm, x), "transform"):
        if x not in ["Compose", "Choose", "Repeat", "RepeatUpTo"]:
            transforms_to_demo.append(x)


audio_path = "../../tests/audio/spk/f10_script4_produced.wav"
signal = AudioSignal(audio_path, offset=6, duration=5)
signal.metadata["loudness"] = AudioSignal(audio_path).ffmpeg_loudness().item()

audio_dict = {
    "Original": {"audio": signal, "spectral_distance": f"{0.0:1.2f}"}
}

distance = metrics.spectral.MelSpectrogramLoss()

for transform_name in transforms_to_demo:
    kwargs = {}
    if transform_name == "BackgroundNoise":
        kwargs["sources"] = ["../../tests/audio/nz"]
    if transform_name == "RoomImpulseResponse":
        kwargs["sources"] = ["../../tests/audio/ir"]
    if transform_name == "CrossTalk":
        kwargs["sources"] = ["../../tests/audio/spk"]
    if "Quantization" in transform_name:
        kwargs["channels"] = ("choice", [8, 16, 32])
    transform_cls = getattr(tfm, transform_name)

    t = transform_cls(prob=1.0, **kwargs)
    t_kwargs = t.instantiate(seed, signal)
    output = t(signal.clone(), **t_kwargs)
    audio_dict[t.name] = {
        "audio": output,
        "spectral_distance": f"{distance(output, signal.clone()).item():1.2f}"
    }

post.disp(audio_dict, first_column="transform")
```

## Introduction

Let's start by looking at the Transforms API. Every transform has two
key functions that the user interacts with:

1. `transform(signal, **kwargs)`: run this to actually transform the input signal using the transform.
2. `instantiate(state, signal)`: run this to instantiate the parameters that a transform requires to run.

Let's look at a concrete example - the LowPass transform. This transform low-passes an AudioSignal so that all energy above a cutoff frequency is deleted. Here's the implementation of it:

```python
class LowPass(BaseTransform):
    def __init__(
        self,
        cutoff: tuple = ("choice", [4000, 8000, 16000]),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)

        self.cutoff = cutoff

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.low_pass(cutoff)
```

First, let's talk about the `_transform` function. It takes two arguments, the `signal`, which is an `AudioSignal` object, and `cutoff` which is a `torch.Tensor`. Note that `signal` may be batched, and `cutoff` may be batched as well. The function just takes the signal and low-passes it, using the low-pass implementation in `core/effects.py`.

Just above `_transform`, we have `_instantiate`, which actually returns a dictionary containing `cutoff` value, which is chosen randomly from a defined distribution. The distribution is defined when you initialize the class, like so:

```{code-cell} ipython3
from audiotools import transforms as tfm

transform = tfm.LowPass()
seed = 0
print(transform.instantiate(seed))
```

Note there's an extra thing: `mask`. Ignore it for now, we'll come back to it later! The instantiated dictionary shows a
single value drawn from the
defined distribution. That distribution chooses from the list `[4000, 8000, 16000]`. We could use a different distribution when we build
our LowPass transform if we wanted:

```{code-cell} ipython3
transform = tfm.LowPass(
    cutoff = ("uniform", 4000, 8000)
)
print(transform.instantiate(seed))
```

This instead draws uniformly between 4000 and 8000 Hz. There's also
a special distribution called `const`, which always returns the same value
(e.g. `(const, 4)` always returns `4`).

Under the hood, `util.sample_from_dist` just calls `state.uniform(4000, 8000)`. Speaking of states, note that it's also passed into `instantiate`. By passing the same seed, you can reliably get the same transform parameters. For example:

```{code-cell} ipython3
transform = tfm.LowPass()
seed = 0
print(transform.instantiate(seed))
```

We see that we got 4000 again for cutoff. Alright, let's apply
our transform to a signal. First, we'll need to construct a signal:

```{code-cell} ipython3
audio_path = "../../tests/audio/spk/f10_script4_produced.wav"
signal = AudioSignal(audio_path, offset=6, duration=5)
```

Okay, let's apply the transform and listen to both:

```{code-cell} ipython3
seed = 0
transform = tfm.LowPass()
kwargs = transform.instantiate(seed)
output = transform(signal.clone(), **kwargs)

# Lines below are to display the audio in a table in the
# notebook.
audio_dict = {
    "signal": signal,
    "low_passed": output,
}
post.disp(audio_dict)
```

And there we have it! Note that we clone the signal before
passing it to the transform:

```python
output = transform(signal.clone(), **kwargs)
```

This is because **signals are changed in-place** by transforms. So you should `clone` the signal before passing it through, if you expect to use the original signal at some point.

Finally, the `keys` attribute of the transform tells you what arguments the transform expects when you run it. For our current transform it's:

```{code-cell} ipython3
print(transform.keys)
```

We see that `cutoff` is expected, and also `mask`. Alright, now we'll explain what that `mask` thing is.

### Masks

Every time you `instantiate` a transform, two things happen:

1. The transforms `_instantiate` is called, initializing the parameters for the transforms (e.g. `cutoff`).
2. The `instantiate` logic in `BaseTransform` is called as well. That logic draws a random number and compares to `transform.prob` to see whether or not the transform should be applied. `prob` is the probability that the transform is applied. It defaults to `1.0` for all transforms. It gets added to the dictionary returned by `instantiate` here:

```python
def instantiate(
    self,
    state: RandomState,
    signal: AudioSignal = None,
):
    ...
    mask = state.rand() <= self.prob
    params[f"mask"] = tt(mask)
    ...
```

Let's set `prob` to `0.5` for our transform, and listen to a few examples, showing the mask along the way:

```{code-cell} ipython3
transform = tfm.LowPass(prob=0.5)
audio_dict = defaultdict(lambda: {})
audio_dict["original"] = {
    "signal": signal,
    "LowPass.cutoff": None,
    "LowPass.mask": None,
}

for seed in range(3):
    kwargs = transform.instantiate(seed)
    output = transform(signal.clone(), **kwargs)

    kwargs = flatten(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            key = ".".join(list(k[-2:]))
            audio_dict[seed][key] = v.item()
    audio_dict[seed]["signal"] = output

post.disp(audio_dict, first_column="seed")
```

The rows where `mask` is `False` have audio that is identical to the
original audio (shown in the top row). Where `mask` is `True`, the transform is applied, as in the last row. The real power of masking comes when you combine it with batching.

### Batching

Let's make a batch of AudioSignals using the `AudioSignal.batch` function. We'll set the batch size to 4:

```{code-cell} ipython3
audio_path = "../../tests/audio/spk/f10_script4_produced.wav"
batch_size = 4
signal = AudioSignal(audio_path, offset=6, duration=5)
signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])
```

Now that we have a batch of signals, let's instantiate a batch of parameters
for the transforms using the `batch_instantiate` function:

```{code-cell} ipython3
transform = tfm.LowPass(prob=0.5)
seeds = range(batch_size)
kwargs = transform.batch_instantiate(seeds)
pp.pprint(kwargs)
```

There are now 4 cutoffs, and 4 mask values in the dictionary, instead of just 1 as before. Under the hood, the `batch_instantiate` function calls `instantiate` with every `seed` in `seeds`, and then collates the results using the `audiotools.util.collate` function. In practice, you'll likely use `audiotools.datasets.AudioDataset` instead to get a single item at a time, and then use the `collate` function as an argument to the torch `DataLoader`'s `collate_fn` argument.

Alright, let's augment the entire batch at once, instead of in a for loop:

```{code-cell} ipython3
transform = tfm.LowPass(prob=0.5)
seeds = range(batch_size)
kwargs = transform.batch_instantiate(seeds)
output_batch = transform(signal_batch.clone(), **kwargs)
```

```{code-cell} ipython3
audio_dict = {}

for i in range(batch_size):
    audio_dict[i] = {
        "input": signal_batch[i],
        "output": output_batch[i],
        "LowPass.cutoff": kwargs["LowPass"]["cutoff"][i].item(),
        "LowPass.mask": kwargs["LowPass"]["mask"][i].item(),
    }

post.disp(audio_dict, first_column="batch_idx")
```

You can see that the masking allows some items in a batch to pass through
the transform unaltered, all in one call.

## Combining transforms

Next, let's see how we can combine transforms.

### The Compose transform

The most common way to combine transforms is to use the `Compose` transform.
`Compose` applies transforms in sequence, and takes a list of transforms as
the first positional argument. `Compose` transforms can be nested as well,
which we'll see later when we start grouping transforms. We'll use another transform (`MuLawQuantization`) to start playing around with `Compose`. Let's build a `Compose` transform that low-passes, then quantizes, and instantiate it:

```{code-cell} ipython3
seed = 0
transform = tfm.Compose(
    [
        tfm.MuLawQuantization(),
        tfm.LowPass(),
    ]
)
kwargs = transform.instantiate(seed)
pp.pprint(kwargs)
```

So, `Compose` instantiated every transform in its list, and put them into the kwargs dictionary. Something else to note: `Compose` also gets a `mask`, just like the other transforms. `Compose` can deal with two transforms of the same type because it just numbers every transform according to their position in the list:

```{code-cell} ipython3
seed = 0
transform = tfm.Compose(
    [
        tfm.LowPass(),
        tfm.LowPass()
    ]
)
kwargs = transform.instantiate(seed)
pp.pprint(kwargs)
```

There are two keys in this dictionary: `0.LowPass`, and `1.LowPass`. Transforms in `Compose` always get a number prefix which corresponds to their position in the sequence of transforms that get applied. The behavior of `Compose` is similar to that of `torch.nn.Sequential`:

```{code-cell} ipython3
net = torch.nn.Sequential(
    torch.nn.Linear(1, 1),
    torch.nn.Linear(1, 1),
)
pp.pprint(net.state_dict())
```

Okay, let's apply the `Compose` transform, just like how we applied the previous
transform:

```{code-cell} ipython3
transform = tfm.Compose(
    [
        tfm.MuLawQuantization(),
        tfm.LowPass(),
    ]
)
seeds = range(batch_size)
kwargs = transform.batch_instantiate(seeds)
output_batch = transform(signal_batch.clone(), **kwargs)
```

```{code-cell} ipython3
audio_dict = make_dict(signal_batch, output_batch, kwargs)
post.disp(audio_dict, first_column="batch_idx")
```

The two transforms were applied in sequence. We can do some pretty crazy stuff
here already, like probabilistically applying just one, both, or none of the
transforms:

```python
transform = tfm.Compose(
    [
        tfm.MuLawQuantization(prob=0.5),
        tfm.LowPass(prob=0.5),
    ],
    prob=0.5
)
```

The masks will get applied in sequence, winnowing down what gets applied.

### Grouping, naming, filtering transforms

To make things a bit easier to handle, we can also explicitly name transforms, group transforms via nesting `Compose` transforms, and filter the application
of transforms by the specified names. Here's an example:

```{code-cell} ipython3
group_a = tfm.Compose(
    [
        tfm.MuLawQuantization(),
        tfm.LowPass(),
    ],
    name="first",
)

group_b = tfm.Compose(
    [
        tfm.VolumeChange(),
        tfm.HighPass(),
    ],
    name="second",
)
transform = tfm.Compose([group_a, group_b])
seeds = range(batch_size)
kwargs = transform.batch_instantiate(seeds)
```

The following applies both sets of transforms in sequence:

```{code-cell} ipython3
output_batch = transform(signal_batch.clone(), **kwargs)
```

But we can also filter for the two specific groups like so:

#### Just first transform

```{code-cell} ipython3
with transform.filter("first"):
    output_batch = transform(signal_batch.clone(), **kwargs)

audio_dict = make_dict(signal_batch, output_batch)
post.disp(audio_dict, first_column="batch_idx")
```

These outputs are low-passed and quantized.

#### Just second transform

```{code-cell} ipython3
with transform.filter("second"):
    output_batch = transform(signal_batch.clone(), **kwargs)

audio_dict = make_dict(signal_batch, output_batch)
post.disp(audio_dict, first_column="batch_idx")
```

These outputs are high-passed and their volume changes.

### The Choose transform

There is also the `Choose` transform which instead of applying all the
transforms in sequence, it instead chooses just one of the transforms
to apply. The following will *either* high-pass or low-pass the entire batch.

```{code-cell} ipython3
transform = tfm.Choose(
    [
        tfm.HighPass(),
        tfm.LowPass(),
    ],
)
seeds = range(batch_size)
kwargs = transform.batch_instantiate(seeds)
output_batch = transform(signal_batch.clone(), **kwargs)

audio_dict = make_dict(signal_batch, output_batch)
post.disp(audio_dict, first_column="batch_idx")
```

All the audio is low-passed. We can flip the order, keeping the same seeds and get the high-pass path.

```{code-cell} ipython3
transform = tfm.Choose(
    [
        tfm.LowPass(),
        tfm.HighPass(),
    ],
)
kwargs = transform.batch_instantiate(seeds)
output_batch = transform(signal_batch.clone(), **kwargs)

audio_dict = make_dict(signal_batch, output_batch)
post.disp(audio_dict, first_column="batch_idx")
```

## Implementing your own transform

You can implement your own transform by doing three things:

1. Implement the init function which takes `prob`, and `name`, and any args with any default distributions you want.
2. Implement the `_instantiate` function to instantiate values for the expected keys.
3. Implement the `_transform` function which takes a `signal` in the first argument, and then other keyword arguments, and does something to the signal and returns a new signal.

Here's a template:

```python
class YourTransform(BaseTransform):
    # Step 1. Define the arguments and their default distribution.
    def __init__(
        self,
        your_arg: tuple = ("uniform", 0.0, 0.1),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.your_arg = your_arg

    def _instantiate(self, state: RandomState):
        # Step 2. Initialize the argument using the distribution
        # or whatever other logic you want to implement here.
        return {"your_arg": util.sample_from_dist(self.your_arg, state)}

    def _transform(self, signal, your_arg):
        # Step 2. Manipulate the signal based on the values
        # passed to this function.
        return do_something(signal, your_arg)
```

## Transforms that require data

There are two transforms which require a dataset to run. They are:

1. `BackgroundNoise`: takes a `sources` argument which points to a list of folders that it can load background noise from.
2. `RoomImpulseResponse`: takes a `sources` argument which points to a list of folders that it can load impulse response data from.

Both of these transforms require an additional argument to their `instantiate` function: an `AudioSignal` object. They get instantiated like this:

```python
seed = ...
signal = ...
transform = tfm.BackgroundNoise(sources=["noise_folder"])
transform.instantiate(seed, signal)
```

The signal is used to load audio from the `sources` that is at the same
sample rate, the same number of channels, and (in the case of `BackgroundNoise`) the same duration as that of `signal`.

## Complete example

Finally, here's a complete example of an entire transform pipeline, which
implements a thorough room simulator.

```{code-cell} ipython3
:tags: ["output_scroll"]
from pathlib import Path
audio_path = "../../tests/audio/spk/f10_script4_produced.wav"
signal = AudioSignal(audio_path, offset=6, duration=5)
batch_size = 10

# Make it into a batch
signal = AudioSignal.batch([signal.clone() for _ in range(batch_size)])

# Create each group of transforms
preprocess = tfm.VolumeChange(name="pre")
process = tfm.Compose(
    [
        tfm.RoomImpulseResponse(sources=["../../tests/audio/ir"]),
        tfm.BackgroundNoise(sources=["../../tests/audio/nz"]),
        tfm.ClippingDistortion(),
        tfm.MuLawQuantization(),
        tfm.LowPass(prob=0.5),
    ],
    name="process",
    prob=0.9,
)
postprocess = tfm.RescaleAudio(name="post")

# Create transform
transform = tfm.Compose([
    preprocess,
    process,
    postprocess,
])

# Instantiate transform (passing in signal because
# some transforms require data).
states = range(batch_size)
kwargs = transform.batch_instantiate(states, signal)

# Apply pre, process, and post to signal in sequence.
output = transform(signal.clone(), **kwargs)

# Apply only pre and post to signal in sequence, skipping process.
with transform.filter("pre", "post"):
    target = transform(signal.clone(), **kwargs)

audio_dict = make_dict(target, output, kwargs)
post.disp(audio_dict, first_column="batch_idx")
```
