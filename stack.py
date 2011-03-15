# Standard library import
import cPickle as pickle
import os.path

# Third-party imports
import theano
from theano import tensor

# Local imports
from .base import Block

class StackedBlocks(Block):
    """
    A stack of Blocks, where the output of a block is the input of the next.
    """
    def __init__(self, layers):
        '''
        Build a stack of layers.

        :type layers: a list of Blocks
        :param layers: the layers to be stacked,
            ordered from bottom (input) to top (output)
        '''
        self.layers = layers
        # Do not duplicate the parameters if some are shared between layers
        self._params = set([p for p in l.params()
                              for l in self.layers])

    def __len__(self):
        return len(self.layers)

    def __call__(self, inputs, repr_indices=-1):
        '''
        Return the output representation of specified layers.

        :param inputs: inputs of the stack

        :type repr_indices: int, or list of ints
        :param repr_indices: Indices of the hidden representations to return.
            0 means the input, -1 the last output.

        :returns: A symbolic variable, or list of symbolic variables,
            containing the requested hidden representations.
        '''
        # Build the hidden representation at each layer
        repr = [inputs]

        for layer in self.layers:
            outputs = layer(repr[-1])
            repr.append(outputs)

        return repr[repr_indices]

    def function(self, name=None, repr_indices=-1):
        inputs = tensor.matrix()
        return theano.function(
                [inputs],
                outputs=self(inputs, repr_indices=repr_indices),
                name=name)

