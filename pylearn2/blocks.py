"""
Feedforward processing objects. Similar to MLP layers, but specialized
for operation on design matrices rather than generic Spaces, and without
a concept of parameters.
"""
# Standard library imports
from __future__ import print_function

import warnings

# Third-party imports
import theano
from theano import tensor
try:
    from theano.sparse import SparseType
except ImportError:
    warnings.warn("Could not import theano.sparse.SparseType")
from theano.compile.mode import get_default_mode

theano.config.warn.sum_div_dimshuffle_bug = False

use_slow_rng = 0
if use_slow_rng:
    print('WARNING: using SLOW rng')
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Block(object):
    """
    Basic building block that represents a simple transformation. By chaining
    Blocks together we can represent complex feed-forward transformations.
    """
    # TODO: Give this input and output spaces to make it different from a
    #       theano Op. Supporting CompositeSpace would allow more complicated
    #       structures than just chains.
    def __init__(self):
        super(Block, self).__init__()
        self.fn = None
        self.cpu_only = False

    def __call__(self, inputs):
        """
        .. todo::

            WRITEME

        * What does this function do?
        * How should inputs be formatted? is it a single tensor, a list of
          tensors, a tuple of tensors?
        """
        raise NotImplementedError(str(type(self)) + 'does not implement ' +
                                  'Block.__call__')

    def function(self, name=None):
        """
        Returns a compiled theano function to compute a representation

        Parameters
        ----------
        name : string, optional
            name of the function
        """
        inputs = tensor.matrix()
        if self.cpu_only:
            return theano.function([inputs], self(inputs), name=name,
                                   mode=get_default_mode().excluding('gpu'))
        else:
            return theano.function([inputs], self(inputs), name=name)

    def perform(self, X):
        """
        .. todo::

            WRITEME
        """
        if self.fn is None:
            self.fn = self.function("perform")
        return self.fn(X)

    def inverse(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(
            "%s does not implement set_input_space yet" % str(type(self)))

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(
            "%s does not implement get_input_space yet" % str(type(self)))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(
            "%s does not implement get_output_space yet" % str(type(self)))


class StackedBlocks(Block):
    """
    A stack of Blocks, where the output of a block is the input of the next.

    Parameters
    ----------
    layers : list of Blocks
        The layers to be stacked, ordered from bottom (input) to top
        (output)
    """

    def __init__(self, layers):
        super(StackedBlocks, self).__init__()

        self._layers = layers
        self._params = set()
        for l in self._layers:
            if not hasattr(l, '_params'):
                self._params = None
                break
            else:
                # Do not duplicate the parameters if some are shared
                # between layers
                self._params.update(l._params)

    def layers(self):
        """
        .. todo::

            WRITEME
        """
        return list(self._layers)

    def __len__(self):
        """
        .. todo::

            WRITEME
        """
        return len(self._layers)

    def __call__(self, inputs):
        """
        Return the output representation of all layers, including the inputs.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with
            the first dimension indexing training examples and the
            second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            A list of theano symbolic (or list thereof), each
            containing the representation at one level.
            The first element is the input.
        """
        # Build the hidden representation at each layer
        repr = [inputs]

        for layer in self._layers:
            outputs = layer(repr[-1])
            repr.append(outputs)

        return repr

    def function(self, name=None, repr_index=-1, sparse_input=False):
        """
        Compile a function computing representations on given layers.

        Parameters
        ----------
        name : string, optional
            name of the function
        repr_index : int, optional
            Index of the hidden representation to return.
            0 means the input, -1 the last output.
        sparse_input : bool, optional
            WRITEME

        Returns
        -------
        WRITEME
        """

        if sparse_input:
            inputs = SparseType('csr', dtype=theano.config.floatX)()
        else:
            inputs = tensor.matrix()

        return theano.function(
            [inputs],
            outputs=self(inputs)[repr_index],
            name=name)

    def concat(self, name=None, start_index=-1, end_index=None):
        """
        Compile a function concatenating representations on given layers.

        Parameters
        ----------
        name : string, optional
            name of the function
        start_index : int, optional
            Index of the hidden representation to start the concatenation.
            0 means the input, -1 the last output.
        end_index : int, optional
            Index of the hidden representation from which to stop
            the concatenation. We must have start_index < end_index.

        Returns
        -------
        WRITEME
        """
        inputs = tensor.matrix()
        return theano.function(
            [inputs],
            outputs=tensor.concatenate(
                self(inputs)[start_index:end_index]),
            name=name)

    def append(self, layer):
        """
        Add a new layer on top of the last one

        Parameters
        ----------
        layer : WRITEME
        """
        self._layers.append(layer)
        if self._params is not None:
            self._params.update(layer._params)

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return self._layers[0].get_input_space()

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self._layers[-1].get_output_space()

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        for layer in self._layers:
            layer.set_input_space(space)
            space = layer.get_output_space()
