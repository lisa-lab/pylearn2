"""Base class for the components in other modules."""
# Standard library imports
import cPickle
import os.path

# Third-party imports
import theano
from theano import tensor

# Local imports
from .utils import sharedX

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

    def save(self, save_dir, save_file):
        """
        Dumps the entire object to a pickle file.
        Individual classes should override __getstate__ and __setstate__
        to deal with object versioning in the case of API changes.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise IOError('save_dir %s is not a directory' % save_dir)
        else:
            fhandle = open(os.path.join(save_dir, save_file), 'w')
            cPickle.dump(self, fhandle, -1)
            fhandle.close()

    @classmethod
    def load(cls, load_dir, load_file):
        """Load a serialized block."""
        filename = os.path.join(load_dir, load_file)
        if not os.path.isfile(filename):
            raise IOError('File %s does not exist' % filename)
        obj = cPickle.load(open(filename))
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError('unpickled object was of wrong class: %s' %
                            obj.__class__)
    
    def function(self, name=None):
        """ Returns a compiled theano function to compute a representation """
        inputs = tensor.matrix()
        return theano.function([inputs], self(inputs), name=name)

class Optimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """
    def updates(self):
        """Return symbolic updates to apply."""
        raise NotImplementedError()

