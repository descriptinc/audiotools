import os
import tempfile

from audiotools.ml import Experiment


def test_experiment():
    with tempfile.TemporaryDirectory() as d:
        with Experiment(d) as exp:
            exp.snapshot()
