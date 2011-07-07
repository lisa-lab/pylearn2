import numpy as N
from theano import config, shared
floatX = config.floatX

class NormalizedEBM:
    """ An Energy-Based Model with an additional parameter representing log Z.
        In practice, this parameter is only approximately correct, though it
        can be learned through methods such as noise-contrastive estimation. """

    def __init__(self, ebm, init_logZ, learn_logZ):
        self.ebm = ebm
        self.logZ = shared(N.cast[floatX](init_logZ))
        self.learn_logZ = learn_logZ

        self.batches_seen = 0
        self.examples_seen = 0
    #

    def log_prob(self, X):
        return - self.ebm.free_energy(X) - self.logZ
    #

    def get_params(self):
        params = self.ebm.get_params()
        if self.learn_logZ:
            params.append(self.logZ)
        #
        return params
    #

    def censor_updates(self, updates):
        self.ebm.censor_updates(updates)
    #

    def redo_theano(self):
        self.ebm.redo_theano()
        self.E_X_batch_func = self.ebm.E_X_batch_func
    #
#
