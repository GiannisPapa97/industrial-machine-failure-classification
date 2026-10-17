"""Fixtures shared by the unit tests.

With ``--import-mode=importlib`` test files cannot import helpers from each other,
so anything they share lives here as a fixture.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def make_xy() -> MakeXY:
    """A factory of synthetic features and a failure target, with the real columns.

    ``make_xy(n, failure_rate, signal, index_offset)`` returns ``(X, y)`` where the
    first ``n * failure_rate`` rows are failures, and failures have a torque higher
    by ``signal`` on average (the larger, the easier to tell them apart). A nonzero
    ``index_offset`` gives both frames the index ``offset, offset + 1, ...`` instead
    of ``0, 1, ...``. The data is the same every time for the same arguments.
    """

    def factory(
        n: int = 600,
        failure_rate: float = 0.1,
        signal: float = 20.0,
        index_offset: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(0)
        n_failures = int(n * failure_rate)
        y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
        X = pd.DataFrame(
            {
                "Type": rng.choice(["L", "M", "H"], size=n),
                "Air temperature": rng.normal(300, 2, n),
                "Process temperature": rng.normal(310, 1.5, n),
                "Rotational speed": rng.integers(1200, 2800, n),
                "Torque": rng.normal(40, 5, n) + y.to_numpy() * signal,
                "Tool wear": rng.integers(0, 250, n),
            }
        )
        if index_offset:
            X.index = pd.Index(np.arange(n) + index_offset)
            y = y.set_axis(X.index)
        return X, y

    return factory
