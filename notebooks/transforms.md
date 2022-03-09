# Transforms

<!-- ## Running this notebook

This notebook creates a model card for a specified model checkpoint. To run
this notebook, you must ensure that `pandoc` and `codebraid` are installed:

```
# https://pandoc.org/installing.html#linux
pip install codebraid
```

The notebook can be run and its output can be copy/pasted to Discourse via:

```
python -m audiotools.post --discourse notebooks/transforms.md > notebooks/transforms.exec.md
```

The contents of `fuzziness.exec.md` can then be copy-pasted to Discourse.
You can also view the contents without uploading to Discourse by outputting to HTML:

```
python -m audiotools.post notebooks/transforms.md > notebooks/transforms.html
```

Which you can then open in a browser to view. -->

```{.python .cb.nb show=code:none+rich_output+stdout:raw+stderr jupyter_kernel=python3}
from audiotools import AudioSignal
from audiotools import post, util
import audiotools.data.transforms as tfm
from audiotools.data import preprocess
from flatten_dict import flatten
import torch

audio_path = "tests/audio/spk/f10_script4_produced.wav"
signal = AudioSignal(audio_path, offset=10, duration=2)

preprocess.create_csv(
    util.find_audio("tests/audio/nz"),
    "/tmp/noises.csv"
)
preprocess.create_csv(
    util.find_audio("tests/audio/ir"),
    "/tmp/irs.csv"
)

transform = tfm.Compose([
    # These transforms get applied to both the input and target
    # audio, so we'll group them here. These are "pre-processing"
    # transforms.
    tfm.Compose([
        tfm.Silence(prob=0.1),
        tfm.VolumeChange(),
    ]),
    tfm.LowPass(),
    tfm.RoomImpulseResponse(csv_files=["/tmp/irs.csv"]),
    tfm.BackgroundNoise(csv_files=["/tmp/noises.csv"]),
    tfm.ClippingDistortion(),
    tfm.MuLawQuantization(),
])

outputs = {}

for seed in range(10):
    output = {}

    kwargs = transform.instantiate(seed, signal)

    noisy = transform(signal.clone(), **kwargs)
    # Only apply the pre-processing transforms to the target.
    target = transform[0](signal.clone(), **kwargs["Compose"])

    output["input"] = target
    params = flatten(kwargs)
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            key = ".".join(list(k[-2:]))
            output[key] = v

    output["output"] = noisy
    outputs[seed] = output

post.disp(outputs, first_column="seed")
```
