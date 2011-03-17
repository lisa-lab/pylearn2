"""Base class for the components in other modules."""
# Standard library imports
import inspect

# Standard library imports
import cPickle
import os.path

# Third-party imports
import theano
from theano import tensor

# Local imports
from .utils import sharedX, subdict

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Block(object):
    """
    Basic building block for deep architectures.
    """
    def params(self):
        """
        Returns a list of *shared* learnable parameters that
        are, in your judgment, typically learned in this
        model.
        """
        # NOTE: We return list(self._params) rather than self._params
        # in order to explicitly make a copy, so that the list isn't
        # absentmindedly modified. If a user really knows what they're
        # doing they can modify self._params.
        return list(self._params)

    def __call__(self, inputs):
        raise NotImplementedError('__call__')

    def save(self, save_file):
        """
        Dumps the entire object to a pickle file.
        Individual classes should override __getstate__ and __setstate__
        to deal with object versioning in the case of API changes.
        """
        save_dir = os.path.dirname(save_file)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fhandle = open(save_file, 'w')
        cPickle.dump(self, fhandle, -1)
        fhandle.close()

    @classmethod
    def fromdict(cls, conf):
        """ Alternative way to build a block, by using a dictionary """
        return cls(**subdict(conf, inspect.getargspec(cls.__init__)[0]))

    @classmethod
    def load(cls, load_file):
        """Load a serialized block."""
        if not os.path.isfile(load_file):
            raise IOError('File %s does not exist' % load_file)
        obj = cPickle.load(open(load_file))
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError('unpickled object was of wrong class: %s' %
                            obj.__class__)

    def function(self, name=None):
        """ Returns a compiled theano function to compute a representation """
        inputs = tensor.matrix()
        return theano.function([inputs], self(inputs), name=name)


class StackedBlocks(Block):
    """
    A stack of Blocks, where the output of a block is the input of the next.
    """
    def __init__(self, layers):
        """
        Build a stack of layers.

        :type layers: a list of Blocks
        :param layers: the layers to be stacked,
            ordered from bottom (input) to top (output)
        """
        self._layers = layers
        # Do not duplicate the parameters if some are shared between layers
        self._params = set([p for l in self._layers for p in l.params()])

    def layers(self):
        return list(self._layers)

    def __len__(self):
        return len(self._layers)

    def __call__(self, inputs):
        """
        Return the output representation of all layers, including the inputs.

        :param inputs: inputs of the stack

        :returns: A list of symbolic variables, each containing the
            representation at one level. The first element is the input.
        """
        # Build the hidden representation at each layer
        repr = [inputs]

        for layer in self._layers:
            outputs = layer(repr[-1])
            repr.append(outputs)

        return repr

    def function(self, name=None, repr_indices=-1):
        """
        Compile a function computing representations on given layers.

        :type name: string
        :param name: name of the function

        :type repr_indices: int, or list of ints
        :param repr_indices: Indices of the hidden representations to return.
            0 means the input, -1 the last output.
        """

        inputs = tensor.matrix()
        return theano.function(
                [inputs],
                outputs=self(inputs)[repr_indices],
                name=name)

    def append(self, layer):
        """
        Add a new layer on top of the last one
        """
        self.layers.append(layer)
        self._params.update(layer.params())


class Optimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """
    def updates(self):
        """Return symbolic updates to apply."""
        raise NotImplementedError()
