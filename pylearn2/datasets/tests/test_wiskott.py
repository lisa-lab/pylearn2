import numpy as np
import unittest

from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.wiskott import Wiskott


def test_wiskott():
    data = Wiskott()
    assert not np.any(np.isinf(data.X))
    assert not np.any(np.isnan(data.X))
