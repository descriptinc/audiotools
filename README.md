# AudioTools

Object-oriented handling of audio signals, with fast augmentation routines, batching, padding, and more.

## Installation
```
pip install git+https://github.com/descriptinc/audiotools
```
OR
```bash
git clone https://github.com/descriptinc/audiotools
cd audiotools
pip install .
```

## Documentation

For documentation, see [the docs](https://descriptinc.github.io/audiotools/).

### Deploying documentation

To build the documentation, do:

```bash
cd docs/
make html
open _build/html/index.html
```

Once you're satisfied with your docs, push them to the gh-pages branch via

```bash
cd docs
bash publish_docs.sh
```

## Quickstart

```python
import audiotools
from audiotools import AudioSignal

signal = AudioSignal("tests/audio/spk/f10_script4_produced.wav", offset=5, duration=5)
signal.play() # Play back the signal in your terminal using ffplay

signal.low_pass(8000) # Low-pass the signal
signal.play() # Play back the low-passed version of the signal
```

For more, see the [documentation](##Documentation).

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
