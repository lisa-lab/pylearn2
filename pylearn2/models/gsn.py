"""
Generative Stochastic Networks

This is described in:
- "Generalized Denoising Auto-Encoders as Generative Models" Bengio, Yao, Alain,
   Vincent. arXiv:1305.6663
- "Deep Generative Stochastic Networks Trainable by Backprop" Bengio,
   Thibodeau-Laufer. arXiv:1306.1091
"""
__authors__ = "Eric Martin"
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"

import functools
import warnings

import theano.tensor as T

from pylearn2.base import StackedBlocks
from pylearn2.models.model import Model

class GSN(StackedBlocks, Model):
    def __init__(self, autoencoders, preact_cors=None, postact_cors=None):
        """
        Initialize an Generative Stochastic Network (GSN) object.

        Parameters
        ----------
        autoencoders : list
            A list of autoencoder objects. As of now, only the functionality
            from the base Autoencoder class is used.
        preact_cor : list
            A list of length len(autoencoders) + 1 where each element is a
            callable (which includes Corruptor objects). The callable at index
            i is called before activating the ith layer. Name stands for
            "preactivation corruptors".
        postact_cor : list
            A list of length len(autoencoders) + 1 where each element is a
            callable (which includes Corruptor objects). The callable at index
            i is called directly after activating the ith layer. Name stands for
            "postactivation corruptors".
        """
        super(StackedBlocks, self).__init__(autoencoders)

        self.preact_cors = preact_cors
        self.postact_cors = postact_cors

        # only for convenience
        self.aes = self._layers

        for i, ae in enumerate(self.aes):
            assert ae.tied_weight, "Autoencoder weights must be tied"

            if i != 0:
                if not (ae.nvis == 0 or ae.nvis == self.aes[i - 1].nhid):
                    warnings.warn("Overwriting number of hidden units so that" +
                                  "layers are the correct size")

                # ensure correct size
                ae.set_visible_size(autoencoder[i - 1].nhid)

        # each row of an activation is the activation of a cell, there is a row
        # for each activation in the mini batch
        # note there are len(self) + 1 activations (one for the visible layer)
        # initialize activations to all zeros

        # FIXME: make right data type
        self.activations = [self.aes[0].nvis] + [ae.nhid for ae in self.aes]

        def _make_callable_list(previous):
            identity = lambda x: x
            if previous is None:
                previous = [identity] * len(self.activations)

            assert len(previous) == len(self.activations)

            for i, f in enumerate(previous):
                if not callable(f):
                    previous[i] = identity
            return previous

        self.preact_cors = _make_callable_list(self.preact_cors)
        self.postact_cors = _make_callable_list(self.postact_cors)

    @functools.wraps(Model.get_params)
    def get_params(self):
        return reduce(lambda a, b: a.extend(b),
                      [ae.get_params() for ae in self.aes],
                      [])

    def get_samples(self, minibatch, walkback=0):
        # FIXME: put minibatch in activations[0]
        activations[0] = minibatch

        reconstructions = [None] * (len(self.aes) + walkback)
        for i in xrange(len(self.aes) + walkback):
            self._update()
            reconstructions[i] = self.activations[0]
        return reconstructions

    def _update(self):
        # odd activations
        self._update_activations(xrange(1, len(activations), 2))

        # even activations
        self._update_activations(xrange(0, len(activations), 2))

    def _update_activations(idx_iter):
        # FIXME: should my lambda use get_weight instead of just .weight?

        from_above = lambda i: (self.aes[i].hidbias +
                                T.dot(self.activations[i + 1],
                                      self.aes[i].weights.T))

        ''' equivalent
        from_below = lambda i: self.aes[i - 1]._hidden_input(
            self.activations[i - 1])
        '''
        from_below = lambda i: (self.aes[i - 1].hidbias +
                                T.dot(self.activations[i - 1],
                                     self.aes[i - 1].weights))

        for i in idx_iter:
            # first compute then hidden activation
            if i == 0:
                self.activations[i] = from_above(i)
            elif i == len(self.activations) - 1:
                self.activations[i] = from_below(i)
            else:
                self.activations[i] = from_below(i) + from_above(i)

            # then activate!

            self.activations[i] = self.preact_cors[i](self.activations[i])

            # using activation function from lower autoencoder
            # no activation function on input layer
            if i != 0 and self.aes[i - 1].act_enc is not None:
                self.activations[i] = self.aes[i - 1].act_enc(self.activations[i])

            self.activations[i] = self.postact_cors[i](self.activations[i])

    def __call__(self, inputs):
        if isinstance(inputs, T.Variable):
            return self.run(inputs)[-1]
        else:
            return [self(i) for i in inputs]
