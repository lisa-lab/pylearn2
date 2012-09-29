import numpy as N
from theano import config, shared
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"


class NormalizedEBM(object):
    """
    An Energy-Based Model with an additional parameter representing log Z.
    In practice, this parameter is only approximately correct, though it
    can be learned through methods such as noise-contrastive estimation.
    """
    def __init__(self, ebm, init_logZ, learn_logZ, logZ_lr_scale=1.0):
        self.ebm = ebm
        self.logZ_driver = shared(N.cast[config.floatX](
            init_logZ / logZ_lr_scale
        ))
        self.learn_logZ = learn_logZ
        self.logZ_lr_scale = logZ_lr_scale

        self.batches_seen = 0
        self.examples_seen = 0

    def log_prob(self, X):
        return -self.ebm.free_energy(X) - self.logZ_driver * self.logZ_lr_scale

    def free_energy(self, X):
        return self.ebm.free_energy(X)

    def get_params(self):
        params = self.ebm.get_params()
        if self.learn_logZ:
            params.append(self.logZ_driver)
        return params

    def censor_updates(self, updates):
        self.ebm.censor_updates(updates)

    def redo_theano(self):
        self.ebm.redo_theano()
        self.E_X_batch_func = self.ebm.E_X_batch_func

    def get_weights(self):
        return self.ebm.get_weights()

    def get_weights_format(self):
        return self.ebm.get_weights_format()
