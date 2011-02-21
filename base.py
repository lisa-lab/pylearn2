"""Base class for the components in other modules."""
import cPickle
import os.path
from itertools import izip

import theano
from theano import tensor

#from pylearn.gd.sgd import sgd_updates
#from pylearn.algorithms.mcRBM import contrastive_cost, contrastive_grad
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
            os.mkdir(save_dir)
        elif not os.path.isdir(save_dir):
            raise IOError('save_dir %s is not a directory' % save_dir)
        else:
            fhandle = open(os.path.join(save_dir, save_file), 'w')
            cPickle.dump(self, fhandle, -1)

    @classmethod
    def load(cls, save_file):
        """Load a serialized block."""
        obj = cPickle.load(open(save_file))
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError('unpickled object was of wrong class: %s' %
                            obj.__class__)

class Trainer(object):
    """
    Basic abstract class for training
    """
    def updates(self):
        """Do one step of training."""
        raise NotImplementedError()

    def function(self):
        """Return a compiled Theano function for training"""
        raise NotImplementedError()

    def save(self, save_dir, save_filename):
        raise NotImplementedError('save')
