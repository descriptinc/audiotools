---
title: AudioTools
---

# AudioTools

Object-oriented handling of audio signals, with fast augmentation routines, batching, padding, and more.


### Install hooks

First install the pre-commit util:

https://pre-commit.com/#install

    pip install pre-commit  # with pip
    brew install pre-commit  # on Mac

Then install the git hooks

    pre-commit install
    # check .pre-commit-config.yaml for details of hooks

Upon `git commit`, the pre-commit hooks will be run automatically on the stage files (i.e. added by `git add`)

**N.B. By default, pre-commit checks only run on staged files**

If you need to run it on all files:

    pre-commit run --all-files

# Feature tour

This README can be run, and then copy-pasted into Discourse to hear all the audio output by installing [codebraid](https://github.com/gpoore/codebraid). To build the README as a standalone HTML:

```
python -m audiotools.post README.md > README.html
```

To build the README into something you can copy/paste
into Discourse:

```
python -m audiotools.post --discourse README.md | pbcopy
```

And then you can paste the output to Discourse (already done [here](https://research.descript.com/t/audiotools-readme-with-audio-examples/562)).


```{.python .cb.nb jupyter_kernel=python3}
import torch

import audiotools
from audiotools import AudioSignal
from audiotools import post
import rich
import matplotlib.pyplot as plt

state = audiotools.util.random_state(0)

spk = AudioSignal("tests/audio/spk/f10_script4_produced.wav", offset=5, duration=5)
ir = AudioSignal("tests/audio/ir/h179_Bar_1txts.wav")
nz = AudioSignal("tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav")
```

## Mixing
Let's first listen to the clean file (and also upload it to Discourse, a feature we'll be using throughout this post):

```{.python .cb.nb show=code:verbatim+stdout:raw}
post.disp(spk)
```

Let's also visualize it:

```{.python .cb.nb show=code:verbatim+rich_output+stdout:raw+stderr}
fig = plt.figure(figsize=(12, 4))
spk.specshow()
post.disp(fig)
```

Let's mix the speaker with noise at varying SNRs. We'll make a deep copy
before each mix, to preserve the original signal in `spk`, as the `mix` function
is applied in-place.

```{.python .cb.nb show=code:verbatim+rich_output+stdout:raw}
outputs = {}
for snr in [0, 10, 20]:
    output = spk.deepcopy().mix(nz, snr=snr)
    outputs[f"snr={snr}"] = output
post.disp(outputs)
```

## Batching signals

We can collate a batch together at random offsets from one file, with
the same duration:

```{.python .cb.nb show=code:verbatim+stdout:raw}
batch_size = 16
spk_batch = AudioSignal.batch([
    AudioSignal.excerpt('tests/audio/spk/f10_script4_produced.wav', duration=2, state=state)
    for _ in range(batch_size)
])
print(spk_batch.markdown())
```

We can listen to different items in the batch:

```{.python .cb.nb show=code:verbatim+stdout:raw}
outputs = {}
for idx in [0, 2, 5]:
    output = AudioSignal(spk_batch[idx], spk_batch.sample_rate)
    outputs[f"batch_idx={idx}"] = output
post.disp(outputs)
```

We can mix each item in the batch at a different SNR:

```{.python .cb.nb show=code:verbatim+stdout:raw}
tgt_snr = torch.linspace(-10, 10, batch_size)
spk_plus_nz_batch = spk_batch.deepcopy().mix(nz, snr=tgt_snr)
```

Let's listen to the first and last item in the output:

```{.python .cb.nb show=code:verbatim+stdout:raw}
outputs = {}
for idx in [0, -1]:
    output = AudioSignal(spk_plus_nz_batch[idx], spk_plus_nz_batch.sample_rate)
    outputs[f"batch_idx={idx}"] = output
post.disp(outputs)
```

The first item was mixed at -10 dB SNR, and the last at 10 dB SNR.

## Perceptual loudness
In Descript, we auto-level to -24dB. Now, we can do the same thing
for a batch of audio signals by using an implementation of the same
LUFS algorithm used in FFMPEG. This implementation is fully
differentiable, and so can be computed on the GPU. Let's see
the loudness of each item in our batch.

```{.python .cb.nb show=code:verbatim+stdout:verbatim}
print(spk_batch.loudness())
```

Now, let's auto-level each item in the batch to -24 dB LUFS.

```{.python .cb.nb show=code:verbatim+stdout:verbatim}
output = spk_batch.deepcopy().normalize(-24)
print(output.loudness())
```

Let's make sure the SNR based mixing we did before was actually correct.

```{.python .cb.nb show=code:verbatim+stdout:verbatim}
print(spk_batch.loudness() - nz.loudness())
print(tgt_snr)
```

Fairly close.

## Convolution

Next, let's convolve our speaker with an impulse response, to make it sound like they're in a room.

```{.python .cb.nb show=code:verbatim+stdout:raw}
convolved = spk.deepcopy().convolve(ir)
```

```{.python .cb.nb show=code:none+stdout:raw}
post.disp(convolved)
```

We can convolve every item in the batch with this impulse response.

```{.python .cb.nb show=code:verbatim+stdout:raw}
spk_batch.deepcopy().convolve(ir)
```

Or if we have a batch of impulse responses, we can convolve a batch of speech signals
with the batch of impulse responses.

```{.python .cb.nb show=code:verbatim+stdout:raw}
ir_batch = AudioSignal.batch([
    AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')
    for _ in range(batch_size)
])
spk_batch.deepcopy().convolve(ir_batch)
```

There's also some syntactic sugar for applying convolution.

```{.python .cb.nb show=code:verbatim+stdout:raw}
spk_batch.deepcopy() @ ir_batch # Same as above.
```

## Equalization
Next, let's apply some equalization to the impulse response, to simulate different mic
responses.

First, we need to figure out the number of bands in the EQ.

```{.python .cb.nb show=code:verbatim+stdout:raw}
n_bands = 6
```

Then, let's make a random EQ curve.
The curve is in dB.

```{.python .cb.nb show=code:verbatim+stdout:raw}
curve = -2.5 + 1 * torch.rand(n_bands)
```

Now, apply it to the impulse response.

```{.python .cb.nb show=code:verbatim+stdout:raw}
eq_ir = ir.deepcopy().equalizer(curve)
```

Then convolve with the signal.

```{.python .cb.nb show=code:verbatim+stdout:raw}
output = spk.deepcopy().convolve(eq_ir)
```

```{.python .cb.nb show=code:none+stdout:raw}
post.disp(output)
```

## Pitch shifting and time stretching
Pitch shifting and time stretching can be applied to signals

```{.python .cb.nb show=code:verbatim+stdout:raw}
outputs = {
    "original": spk,
    "pitch_shifted": spk.deepcopy().pitch_shift(2),
    "time_stretched": spk.deepcopy().time_stretch(0.8),
}
post.disp(outputs)
```

Like other transformations, they also get applied
across an entire batch.

```{.python .cb.nb show=code:verbatim+stdout:raw}
spk_batch.deepcopy().pitch_shift(2)
spk_batch.deepcopy().time_stretch(0.8)
```

## Codec transformations
This one is a bit wonky, but you can take audio, and convert it into a
a highly compressed format, and then get the samples back out. This
creates a sort of "Zoom-y" effect.

```{.python .cb.nb show=code:verbatim+stdout:raw}
output = spk.deepcopy().apply_codec("Ogg")
```

```{.python .cb.nb show=code:none+stdout:raw}
post.disp(output)
```

## Putting it all together
This is a fluent interface so things can be chained together easily.
Let's augment an entire batch by chaining these effects together.
We'll start from scratch, loading the batch fresh each time to
avoid overwriting anything inside the augmentation pipeline.

```{.python .cb.nb show=code:verbatim+stdout:raw}
def load_batch(batch_size, state=None):
    spk_batch = AudioSignal.batch([
        AudioSignal.salient_excerpt('tests/audio/spk/f10_script4_produced.wav', duration=5, state=state)
        for _ in range(batch_size)
    ])
    nz_batch = AudioSignal.batch([
        AudioSignal.excerpt('tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav', duration=5, state=state)
        for _ in range(batch_size)
    ])
    ir_batch = AudioSignal.batch([
        AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')
        for _ in range(batch_size)
    ])
    return spk_batch, nz_batch, ir_batch
```

We'll apply the following pipeline, randomly getting parameters for each effect.
1. Pitch shift
2. Time stretch
3. Equalize noise.
4. Equalize impulse response.
5. Convolve speech with impulse response.
6. Mix speech and noise at some random SNR.

```{.python .cb.nb show=code:verbatim+stdout:raw}
batch_size = 4

# Seed is given to function for reproducibility.
def augment(seed):
    state = audiotools.util.random_state(seed)
    spk_batch, nz_batch, ir_batch = load_batch(batch_size, state)

    n_semitones = state.uniform(-2, 2)
    factor = state.uniform(0.8, 1.2)
    snr = state.uniform(10, 40, batch_size)

    # Make a copy so we have it later for training targets.
    clean_spk = spk_batch.deepcopy()

    spk_batch = (
        spk_batch
            .pitch_shift(n_semitones)
            .time_stretch(factor)
    )

    # Augment the noise signal with equalization
    n_bands = 6
    curve = -1 + 1 * state.rand(nz_batch.batch_size, n_bands)
    nz_batch = nz_batch.equalizer(curve)

    # Augment the impulse response to simulate microphone effects.
    n_bands = 6
    curve = -1 + 1 * state.rand(ir_batch.batch_size, n_bands)
    ir_batch = ir_batch.equalizer(curve)

    # Convolve
    noisy_spk = (
        spk_batch
            .convolve(ir_batch)
            .mix(nz_batch, snr=snr)
    )

    return clean_spk, noisy_spk
```

Let's augment and then listen to each item in the batch.

```{.python .cb.nb show=code:verbatim+stdout:raw}
clean_spk, noisy_spk = augment(0)
sr = clean_spk.sample_rate
for i in range(clean_spk.batch_size):
    print(f"**Sample {i+1}**\n")
    outputs = {
        "clean": AudioSignal(clean_spk[i], sr),
        "noisy": AudioSignal(noisy_spk[i], sr),
    }
    post.disp(outputs)
```
