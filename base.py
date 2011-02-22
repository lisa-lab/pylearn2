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
            os.mkdir(save_dir)
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

class Optimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """
    def updates(self):
        """Return symbolic updates to apply"""
        raise NotImplementedError()

    def function(self, inputs):
        """Return a compiled Theano function for training"""
        raise NotImplementedError()

    def learning_rates_setup(self, conf, params):
        """
        Initializes parameter-specific learning rate dictionary and shared
        variables for the annealed base learning rate and iteration number.
        """
        # Take care of learning rate scales for individual parameters
        self.learning_rates = {}

        for parameter in params:
            lr_name = '%s_lr' % parameter.name
            thislr = conf.get(lr_name, 1.)
            self.learning_rates[parameter] = sharedX(thislr, lr_name)

        # A shared variable for storing the iteration number.
        self.iteration = sharedX(theano._asarray(0, dtype='int32'), name='iter')

        # A shared variable for storing the annealed base learning rate, used
        # to lower the learning rate gradually after a certain amount of time.
        self.annealed = sharedX(conf['base_lr'], 'annealed')

    def learning_rate_updates(self, conf, params):
        ups = {}
        # Base learning rate per example.
        base_lr = theano._asarray(self.conf['base_lr'], dtype=floatX)

        # Annealing coefficient. Here we're using a formula of
        # base_lr * min(0.0, max(base_lr, lr_anneal_start / (iteration + 1))
        frac = self.conf['lr_anneal_start'] / (self.iteration + 1.)
        annealed = tensor.clip(
            tensor.cast(frac, floatX),
            0.0,    # minimum learning rate
            base_lr # maximum learning rate
        )

        # Update the shared variable for the annealed learning rate.
        ups[self.annealed] = annealed
        ups[self.iteration] = self.iteration + 1

        # Calculate the learning rates for each parameter, in the order
        # they appear in self.params
        learn_rates = [annealed * self.learning_rates[p] for p in params]
        return ups, learn_rates
