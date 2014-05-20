"""
Cross-validation with MLPs.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

from pylearn2.models.mlp import Layer, PretrainedLayer


class PretrainedLayerCV(Layer):
    """
    Container of PretrainedLayer objects for use with TrainCV.

    Parameters
    ----------
    layer_name: str
        Name of layer.
    layer_content: array_like
        Pretrained layer models for each dataset subset.
    """
    def __init__(self, layer_name, layer_content):
        self.layer_name = layer_name
        self._folds = [PretrainedLayer(layer_name, subset_content)
                       for subset_content in layer_content]

    def select_fold(self, k):
        """
        Choose a single cross-validation fold to represent.

        Parameters
        ----------
        k : int
            Index of selected fold.
        """
        return self._folds[k]

    def set_input_space(self, space):
        """
        Set input space.

        Parameters
        ----------
        space : Space
            The input space for this layer.
        """
        return [fold.set_input_space(space) for fold in self._folds]

    def get_params(self):
        """Get parameters."""
        return self._folds[0].get_params()

    def get_input_space(self):
        """Get input space."""
        return self._folds[0].get_input_space()

    def get_output_space(self):
        """Get output space."""
        return self._folds[0].get_output_space()

    def get_monitoring_channels(self):
        """Get monitoring channels."""
        return self._folds[0].get_monitoring_channels()
