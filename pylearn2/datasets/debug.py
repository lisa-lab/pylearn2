"""
.. todo::

    WRITEME
"""
import numpy as N
from pylearn2.datasets import dense_design_matrix


class DebugDataset(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME
    """

    def __init__(self):
        """
        .. todo::

            WRITEME
        """

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3))

        super(DebugDataset, self).__init__(X=N.asarray([[1.0, 0.0],
                                                        [0.0, 1.0]]),
                                           view_converter=view_converter)

        assert not N.any(N.isnan(self.X))
