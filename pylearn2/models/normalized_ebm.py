"""
The NormalizedEBM class.
"""

import functools
import numpy as np

from theano import config, shared

from pylearn2.models import Model

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"


class NormalizedEBM(Model):
    """
    An Energy-Based Model with an additional parameter representing log Z.

    In practice, this parameter is only approximately correct, though it
    can be learned through methods such as noise-contrastive estimation.

    Parameters
    ----------
    ebm : WRITEME
        The underlying EBM to learn.
    init_logZ : float
        Initial estimate of log Z
    learn_logZ : bool
        If true, learn log Z. Otherwise, continue to assume init_logZ is
        accurate.
    logZ_lr_scale : float
        How much to scale the learning rate on the log Z estimate
    """
    def __init__(self, ebm, init_logZ, learn_logZ, logZ_lr_scale=1.0):
        super(NormalizedEBM, self).__init__()
        self.ebm = ebm
        self.logZ_driver = shared(np.cast[config.floatX](
            init_logZ / logZ_lr_scale
        ))
        self.learn_logZ = learn_logZ
        self.logZ_lr_scale = logZ_lr_scale

        self.batches_seen = 0
        self.examples_seen = 0

    def log_prob(self, X):
        """
        Returns the log probability of a batch of examples.

        Parameters
        ----------
        X : WRITEME
            The examples whose log probability should be computed.

        Returns
        -------
        log_prob : WRITEME
            The log probability of the examples.
        """
        return -self.ebm.free_energy(X) - self.logZ_driver * self.logZ_lr_scale

    def free_energy(self, X):
        """
        Returns the free energy of a batch of examples.

        Parameters
        ----------
        X : WRITEME
            The examples whose free energy should be computed.

        Returns
        -------
        free_energy : WRITEME
            The free energy of the examples.
        """
        return self.ebm.free_energy(X)

    @functools.wraps(Model.get_params)
    def get_params(self):
        params = self.ebm.get_params()
        if self.learn_logZ:
            params.append(self.logZ_driver)
        return params

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        self.ebm.modify_updates(updates)

    @functools.wraps(Model.redo_theano)
    def redo_theano(self):
        self.ebm.redo_theano()
        self.E_X_batch_func = self.ebm.E_X_batch_func

    @functools.wraps(Model.get_weights_format)
    def get_weights(self):
        return self.ebm.get_weights()

    @functools.wraps(Model.get_weights_format)
    def get_weights_format(self):
        return self.ebm.get_weights_format()
