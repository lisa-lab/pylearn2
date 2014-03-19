__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"


import numpy as np
N = np
import os.path
import commands

from pylearn2.datasets import dense_design_matrix


class Wiskott(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME
    """
    def __init__(self):

        pylearn2_data_path = commands.getoutput("echo $PYLEARN2_DATA_PATH")
        _path = "wiskott/wiskott_fish_layer0_15_standard_64x64_shuffled.npy"
        data_path = os.path.join(pylearn2_data_path,_path)

        X = 1. - np.load(data_path)
        view_converter = dense_design_matrix.DefaultViewConverter((64, 64, 1))

        super(Wiskott, self).__init__(X=X, view_converter=view_converter)

        assert not N.any(N.isnan(self.X))
