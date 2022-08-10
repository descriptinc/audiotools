import tempfile

import torch
from torch import nn

from audiotools import ml
from audiotools import util

SEED = 0


def seed_and_run(model, *args, **kwargs):
    util.seed(SEED)
    return model(*args, **kwargs)


class Model(ml.BaseModel):
    def __init__(self, arg1: float = 1.0):
        super().__init__()
        self.arg1 = arg1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class OtherModel(ml.BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_base_model():
    # Save and load
    x = torch.randn(10, 1)
    model1 = Model()

    assert model1.device == torch.device("cpu")

    out1 = seed_and_run(model1, x)

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        model1.save(f.name, extern=["test_model"])
        model2 = Model.load(f.name)
        out2 = seed_and_run(model2, x)
        assert torch.allclose(out1, out2)

        # test re-export
        model2.save(f.name, extern=["test_model"])
        model3 = Model.load(f.name)
        out3 = seed_and_run(model3, x)
        assert torch.allclose(out1, out3)

        # make sure legacy/save load works
        model1.save(f.name, package=False, extern=["test_model"])
        model2 = Model.load(f.name)
        out2 = seed_and_run(model2, x)
        assert torch.allclose(out1, out2)

        # make sure new way -> legacy save -> legacy load works
        model1.save(f.name, package=True, extern=["test_model"])
        model2 = Model.load(f.name)
        model2.save(f.name, package=False, extern=["test_model"])
        model3 = Model.load(f.name)
        out3 = seed_and_run(model3, x)

        # save/load without package, but with model2 being a model
        # without an argument of arg1 to its instantiation.
        model1.save(f.name, package=False, extern=["test_model"])
        model2 = OtherModel.load(f.name)
        out2 = seed_and_run(model2, x)
        assert torch.allclose(out1, out2)

        assert torch.allclose(out1, out3)
