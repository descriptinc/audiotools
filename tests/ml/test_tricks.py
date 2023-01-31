import numpy as np
import torch
from torch import nn

from audiotools import ml
from audiotools.ml.tricks import compute_grad_norm


class DummyModel(ml.BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def test_compute_grad_norm():
    model = DummyModel()
    grad_norm = compute_grad_norm(model)
    assert grad_norm == 0.0

    x, y = torch.randn(1), torch.randn(1)
    y_hat = model(x)
    loss = (y_hat - y).pow(2)
    loss.backward()

    grad_norm = compute_grad_norm(model, mask_nan=True)
    assert grad_norm == (x * 2.0 * (y_hat - y)).abs()


def test_autoclip():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    autoclip = ml.tricks.AutoClip(0)  # min-clip
    grad_history = []

    for _ in range(10):
        x = torch.randn(1)
        y = 10 * x
        y_hat = model(x)

        optimizer.zero_grad()
        loss = (y_hat - y).pow(2)
        loss.backward()
        clip_value, grad_norm = autoclip(model)
        optimizer.step()

        assert grad_norm == (x * 2.0 * (y_hat - y)).abs()
        clipped_grad_norm = compute_grad_norm(model)

        assert clipped_grad_norm <= grad_norm + 1e-6
        assert clipped_grad_norm <= clip_value + 1e-6
        grad_history.append(grad_norm)

    state_dict = autoclip.state_dict()
    assert np.allclose(state_dict["grad_history"], grad_history)
    autoclip.load_state_dict(state_dict)

    autoclip = ml.tricks.AutoClip(0, max_history=100)


def test_autobalance():
    losses = torch.randn(10).abs().tolist()

    ratios = torch.randn(10).abs()
    ratios = torch.nn.functional.normalize(ratios, p=1, dim=-1)
    ratios = ratios.tolist()

    autobalance = ml.tricks.AutoBalance(ratios)
    output = autobalance(*losses)

    assert np.allclose(sum(autobalance.weights), 1.0)
    for w, l, r in zip(autobalance.weights, losses, ratios):
        assert np.allclose(w * l / sum(output), r)

    autobalance = ml.tricks.AutoBalance(ratios, max_iters=5)
    weight_history = []
    for i in range(10):
        losses = torch.randn(10).abs().tolist()
        autobalance(*losses)

        if i > autobalance.max_iters:
            assert autobalance.weights == weight_history[-1]
        else:
            weight_history.append(autobalance.weights)
