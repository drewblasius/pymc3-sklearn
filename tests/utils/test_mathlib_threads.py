import pytest
from sk_pymc3.utils import set_mathlib_threads


def test_set_threads():
    set_mathlib_threads(1)


def test_improper_set_threads():
    with pytest.raises(ValueError):
        set_mathlib_threads(1.3)

    with pytest.raises(ValueError):
        set_mathlib_threads(-1)
