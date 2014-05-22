"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

class Wiskott(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME
    """
    def __init__(self):
        path = "${PYLEARN2_DATA_PATH}/wiskott/wiskott"\
             + "_fish_layer0_15_standard_64x64_shuffled.npy"

        X = 1. - load(path)

        view_converter = dense_design_matrix.DefaultViewConverter((64,64,1))

        super(Wiskott,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
