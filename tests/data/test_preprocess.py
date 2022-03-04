import tempfile
from pathlib import Path

from audiotools.core.util import find_audio
from audiotools.data import preprocess


def test_create_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(
            find_audio("./tests/audio/spk", ext=["wav"]), f.name, loudness=True
        )
