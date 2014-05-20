"""
Cross-validation with blocks.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

from pylearn2.blocks import StackedBlocks


class StackedBlocksCV(object):
    """
    Multi-layer transforms using cross-validation models.

    Parameters
    ----------
    layers : iterable (list of lists)
        Cross-validation models for each layer. Should be a list of lists,
        where the first index is for the layer and the second index is for
        the cross-validation fold.
    """
    def __init__(self, layers):
        stacked_blocks = []
        n_folds = len(layers[0])
        assert all([len(layer) == n_folds for layer in layers])

        # stack the k-th block from each layer
        for k in xrange(n_folds):
            this_blocks = []
            for i, layer in enumerate(layers):
                this_blocks.append(layer[k])
            this_stacked_blocks = StackedBlocks(this_blocks)
            stacked_blocks.append(this_stacked_blocks)

        # _folds contains a StackedBlocks instance for each CV fold
        self._folds = stacked_blocks

    def select_fold(self, k):
        """
        Choose a single cross-validation fold to represent.

        Parameters
        ----------
        k : int
            Index of selected fold.
        """
        return self._folds[k]

    def get_input_space(self):
        """Get input space."""
        return self._folds[0][0].get_input_space()

    def get_output_space(self):
        """Get output space."""
        return self._folds[0][-1].get_output_space()

    def set_input_space(self, space):
        """
        Set input space.

        Parameters
        ----------
        space : WRITEME
            Input space.
        """
        for fold in self._folds:
            this_space = space
            for layer in fold._layers:
                layer.set_input_space(this_space)
                this_space = layer.get_output_space()
