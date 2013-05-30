import numpy as np
from pylearn2.datasets.four_regions import FourRegions


def test_four_regions():
    dataset = FourRegions(5000)
    X = dataset.get_design_matrix()
    np.testing.assert_(((X < 1.) & (X > -1.)).all())
    y = dataset.get_targets()
    np.testing.assert_equal(np.unique(y), [0, 1, 2, 3])
