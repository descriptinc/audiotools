import numpy as np
from audiotools.util import random_state
import pytest

def test_check_random_state():
    # seed is None
    rng_type = type(np.random.RandomState(10))
    rng = random_state(None)
    assert type(rng) == rng_type

    # seed is int
    rng = random_state(10)
    assert type(rng) == rng_type

    # seed is RandomState
    rng_test = np.random.RandomState(10)
    rng = random_state(rng_test)
    assert type(rng) == rng_type

    # seed is none of the above : error
    pytest.raises(ValueError, random_state, 'random')