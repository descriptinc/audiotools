from typing import List

import numpy as np
import torch
from torch import nn


def compute_grad_norm(model, mask_nan: bool = False):
    """Computes the gradient norm of a model after a
    backwards pass.

    Parameters
    ----------
    model : nn.Module
        Model to compute gradient norm of.
    mask_nan : bool, optional
        Whether to mask away NaNs, useful if using
        automatic mixed-precision, by default False

    Returns
    -------
    float
        Gradient norm of model.
    """
    all_norms = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_data = p.grad.data

        if mask_nan:
            nan_mask = torch.isfinite(p.grad.data)
            grad_data = grad_data[nan_mask]
        param_norm = float(grad_data.norm(2))
        all_norms.append(param_norm)

    total_norm = float(torch.tensor(all_norms).norm(2))
    return total_norm


class AutoClip:
    """
    Adds AutoClip during training.
    The gradient is clipped to the percentile'th percentile of
    gradients seen during training. Proposed in [1].

    1.  Prem Seetharaman, Gordon Wichern, Bryan Pardo,
        Jonathan Le Roux. "AutoClip: Adaptive Gradient
        Clipping for Source Separation Networks." 2020
        IEEE 30th International Workshop on Machine
        Learning for Signal Processing (MLSP). IEEE, 2020.

    Parameters
    ----------
    percentile : float, optional
        Percentile to clip gradients to, by default 10
    frequency : int, optional
        How often to re-compute the clipping value.
    """

    def __init__(self, percentile: float = 10, frequency: int = 1, mask_nan: int = 0):
        self.grad_history = []
        self.percentile = percentile
        self.frequency = frequency
        self.mask_nan = bool(mask_nan)

        self.iters = 0

    def state_dict(self):
        state_dict = {
            "grad_history": self.grad_history,
            "percentile": self.percentile,
            "frequency": self.frequency,
            "mask_nan": self.mask_nan,
            "iters": self.iters,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def __call__(self, model):
        if self.iters % self.frequency == 0:
            grad_norm = compute_grad_norm(model, self.mask_nan)
            self.grad_history.append(grad_norm)

        grad_norm = self.grad_history[-1]
        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        self.iters += 1
        return clip_value, grad_norm


class AutoBalance(nn.Module):
    """Auto-balances losses with each other by solving a system of
    equations.
    """

    def __init__(
        self, ratios: List[float] = [1], frequency: int = 1, max_iters: int = None
    ):
        super().__init__()

        self.frequency = frequency
        self.iters = 0
        self.max_iters = max_iters
        self.weights = [1 for _ in range(len(ratios))]

        # Set up the problem
        ratios = torch.from_numpy(np.array(ratios))

        n_losses = ratios.shape[0]

        off_diagonal = torch.eye(n_losses) - 1
        diagonal = (n_losses - 1) * torch.eye(n_losses)

        A = off_diagonal + diagonal
        B = torch.zeros(1 + n_losses)
        B[-1] = 1

        W = 1 / ratios

        self.register_buffer("A", A.float())
        self.register_buffer("B", B.float())
        self.register_buffer("W", W.float())
        self.ratios = ratios

    def __call__(self, *loss_vals):
        exceeded_iters = False
        if self.max_iters is not None:
            exceeded_iters = self.iters >= self.max_iters

        with torch.no_grad():
            if self.iters % self.frequency == 0 and not exceeded_iters:
                num_losses = self.ratios.shape[-1]
                L = self.W.new(loss_vals[:num_losses])
                if L[L > 0].shape == L.shape:
                    _A = self.A * L * self.W
                    _A = torch.vstack([_A, torch.ones_like(self.W)])

                    # Solve with least squares for weights so each
                    # loss function matches what is given in ratios.
                    X = torch.linalg.lstsq(_A.float(), self.B.float(), rcond=None)[0]

                    self.weights = X.tolist()

        self.iters += 1
        return [w * l for w, l in zip(self.weights, loss_vals)]
