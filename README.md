# AudioTools

Object-oriented handling of audio signals, with fast augmentation routines,
batching, padding, and more.


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

This feature tour below comes from `demo.py` in this repository.

```python
import audiotools
from audiotools import AudioSignal
import torch

spk = AudioSignal('tests/audio/spk/f10_script4_produced.wav', offset=5, duration=2)
ir = AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')
nz = AudioSignal('tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav')
```

Mixing
------
Listen to the clean file.

```python
spk.play()
```

Mix speaker with noise at varying SNR.
We make the deep copy before each one to preserve
the original signal.

```python
spk.deepcopy().mix(nz, snr=20).play() # High SNR
spk.deepcopy().mix(nz, snr=10).play() # Medium SNR
spk.deepcopy().mix(nz, snr=0).play() # Low SNR
```

Collate a batch together at random offsets
from one file, same duration.

```python
batch_size = 16
spk_batch = AudioSignal.batch([
    AudioSignal.excerpt('tests/audio/spk/f10_script4_produced.wav', duration=2)
    for _ in range(batch_size)
])
```

Printing gives useful information:

```python
print(spk_batch)
```

```
AudioSignal
Duration: 2.000 sec
Batch size: 16
Path: path unknown
Sample rate: 44100 Hz
Number of channels: 1 ch
STFT Parameters: STFTParams(window_length=2048, hop_length=512, window_type='sqrt_hann')
```

Listen to items in the batch:

```python
spk_batch.play(5) # 5th item
```

Mix each item in the batch at a different SNR:

```python
tgt_snr = torch.linspace(-10, 10, batch_size)
output = spk_batch.deepcopy().mix(nz, snr=tgt_snr)
```

Play the last item:

```python
output.play(-1)
```

Differentiable perceptual loudness
-----------------------
In Descript, we auto-level to -24dB. Now, we can do the same thing
for a batch of audio signals by using an implementation of the same
LUFS algorithm used in FFMPEG. This implementation is fully
differentiable, and so can be computed on the GPU. Let's see
the loudness of each item in our batch.

```python
print(spk_batch.loudness())
```

```
tensor([-16.6756, -18.8920, -14.6954, -15.5525, -17.1260, -18.1961, -17.2073,
        -17.7730, -17.2884, -16.5896, -16.6003, -17.6432, -16.6057, -16.9705,
        -19.3882, -17.8118])
```

Now, let's auto-level each item in the batch to -24 dB LUFS.

```python
output = spk_batch.deepcopy().normalize(-24)
print(output.loudness())
```

```
tensor([-23.9894, -23.9930, -23.9824, -23.9915, -23.9884, -23.9796, -23.9294,
        -23.9888, -23.9596, -23.9882, -23.9893, -23.9739, -23.9899, -23.9931,
        -23.9440, -23.9878])
```

Let's make sure the SNR based mixing we did before was actually correct.

```
print(spk_batch.loudness() - nz.loudness())
print(tgt_snr)
```

```
tensor([-9.7829, -8.3380, -7.3028, -5.9749, -4.6627, -3.3333,  0.0000, -0.6667,
         0.6667,  2.0000,  3.3333,  4.6667,  6.0000,  7.3333,  8.6667, 10.0000])
tensor([-10.0000,  -8.6667,  -7.3333,  -6.0000,  -4.6667,  -3.3333,  -2.0000,
         -0.6667,   0.6667,   2.0000,   3.3333,   4.6667,   6.0000,   7.3333,
          8.6667,  10.0000])
```

Fairly close.

Convolution
-----------
Next, let's convolve our speaker with an impulse response, to make it sound like they're in a room.

```python
spk.deepcopy().convolve(ir)
```

We can convolve every item in the batch with this impulse response.

```python
spk_batch.deepcopy().convolve(ir)
```

Or if we have a batch of impulse responses, we can convolve a batch of speech signals
with the batch of impulse responses.

```python
ir_batch = AudioSignal.batch([
    AudioSignal('tests/audio/ir/h179_Bar_1txts.wav')
    for _ in range(batch_size)
])
spk_batch.deepcopy().convolve(ir_batch)
```

There's also some syntactic sugar for applying convolution.

```python
spk_batch.deepcopy() @ ir_batch # Same as above.
```

Equalization
------------
Next, let's apply some equalization to the impulse response, to simulate different mic
responses.

First, we need to figure out the number of bands in the EQ.

```python
n_bands 6
```

Then, let's make a random EQ curve.
The curve is in dB.

```python
curve = -2.5 + 1 * torch.rand(n_bands)
```

Now, apply it to the impulse response.

```python
eq_ir = ir.deepcopy().equalizer(curve)
```

Then convolve with the signal.

```python
spk.deepcopy().convolve(eq_ir)
```

Pitch shifting and time stretching
----------------------------------
Pitch shifting and time stretching can be applied to signals

```python
spk.deepcopy().pitch_shift(2)
spk.deepcopy().pitch_shift(-2)

spk.deepcopy().time_stretch(1.2)
spk.deepcopy().time_stretch(0.8)
```

Like other transformations, they also get applied
across an entire batch.

```python
spk_batch.deepcopy().pitch_shift(2)
spk_batch.deepcopy().time_stretch(2)
```

Codec transformations
---------------------
This one is a bit wonky, but you can take audio, and convert it into a
a highly compressed format, and then get the samples back out. This
creates a sort of "Zoom-y" effect.

```python
spk.deepcopy().apply_codec("Ogg")
```

Putting it all together
-----------------------
This is a fluent interface so things can be chained together easily.
Let's augment an entire batch by chaining these effects together.
We'll start from scratch, loading the batch fresh each time to
avoid overwriting anything inside the augmentation pipeline.

```python
def load_batch(batch_size, state=None):
    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt('tests/audio/spk/f10_script4_produced.wav', duration=1, state=state)
        for _ in range(batch_size)
    ])
    nz_batch = AudioSignal.batch([
        AudioSignal.excerpt('tests/audio/nz/f5_script2_ipad_balcony1_room_tone.wav', duration=1, state=state)
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

```python
batch_size = 16

# Seed is given to function for reproducibility.
def augment(seed):
    state = audiotools.util.random_state(seed)
    spk_batch, nz_batch, ir_batch = load_batch(batch_size, state)

    n_semitones = state.uniform(-2, 2)
    factor = state.uniform(0.8, 1.2)
    snr = state.uniform(10, 40, batch_size)

    # We're not trying to undo pitch shifting/time streching.
    spk_batch = (
        spk_batch
            .pitch_shift(n_semitones)
            .time_stretch(factor)
    )
    # Make a copy so we have it later for training targets.
    clean_spk = spk_batch.deepcopy()

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

```python
clean_spk, noisy_spk = augment(0)
for i in range(clean_spk.batch_size):
    clean_spk.play(i)
    noisy_spk.play(i)
```
