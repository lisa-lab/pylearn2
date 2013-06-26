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

import numpy as np
import theano.tensor as T

from pylearn2.base import StackedBlocks
from pylearn2.models.autoencoder import Autoencoder
from pylearn2.models.model import Model
from pylearn2.utils import sharedX

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
        super(GSN, self).__init__(autoencoders)

        # only for convenience
        self.aes = self._layers

        for i, ae in enumerate(self.aes):
            assert ae.tied_weights, "Autoencoder weights must be tied"

            if i != 0:
                if ae.weights is None:
                    print 'wtf'
                    ae.set_visible_size(self.aes[i - 1].nhid)
                else:
                    assert (ae.weights.get_value().shape[0] ==
                            self.aes[i - 1].nhid)

        def _make_callable_list(previous):
            num_activations = len(self.aes) + 1
            identity = lambda x: x
            if previous is None:
                previous = [identity] * num_activations

            assert len(previous) == num_activations

            for i, f in enumerate(previous):
                if not callable(f):
                    previous[i] = identity
            return previous

        self.preact_cors = _make_callable_list(preact_cors)
        self.postact_cors = _make_callable_list(postact_cors)

    @classmethod
    def new(cls, layer_sizes, vis_corruptor,
            hidden_pre_corruptor, hidden_post_corruptor, act="sigmoid"):
        """
        This is just a convenience method to initialize GSN instances. The
        __init__ method is far more general, but this should capture most
        of the GSM use cases.

        Parameters
        ----------
        layer_sizes : list of integers
            Each element of this list states the size of a layer in the GSN. The
            first element in this list is the size of the visual layer, which
            can be initially set to 0 and later specified with
            GSN.set_visible_size.
        vis_corruptor : callable
            A callable object (such as a Corruptor) that is used to corrupt the
            visible layer.
        hidden_pre_corruptor : callable
            Same sort of object as the vis_corruptor, used to corrupt (add noise)
            to all of the hidden activations prior to activation.
        hidden_post_corruptor : callable
            Same sort of object as the other corruptors, used to corrupt hidden
            activations after activation.
        act : callable or string
            The value for act must be a valid value for both the act_enc and the
            act_dec arguments in Autoencoder.__init__.
        """
        aes = []
        for i in xrange(len(layer_sizes) - 1):
            aes.append(Autoencoder(layer_sizes[i], layer_sizes[i + 1],
                                   act, act, tied_weights=True))

        pre_corruptors = [vis_corruptor] + [hidden_pre_corruptor] * len(aes)
        post_corruptors = [None] + [hidden_post_corruptor] * len(aes)
        return GSN(aes, preact_cors=pre_corruptors, postact_cors=post_corruptors)

    @functools.wraps(Model.get_params)
    def get_params(self):
        params = []
        for ae in self.aes:
            params.extend(ae.get_params())
        return params

    def set_visible_size(self, *args, **kwargs):
        """
        Proxy method to Autoencoder.set_visible_size. Sets visible size on first
        autoencoder.
        """
        self.aes[0].set_visible_size(*args, **kwargs)

    def _run(self, minibatch, walkback=0):
        """
        This runs the GSN on input 'minibatch' are returns all of the activations
        at every time step.

        Parameters
        ----------
        minibatch : tensor_like
            Theano symbolic representing the input minibatch.
        walkback : int
            How many walkback steps to perform. DOCUMENT BETTER

        Returns
        ---------
        steps : list of list of tensor_likes
            A list of the activations at each time step. The activations
            themselves are lists of tensor_like symbolic (shared) variables.
            A time step consists of a call to the _update function (so updating
            both the odd and even layers).

        Notes
        ---------
            At the beginning of execution, the activations in the higher layers
            are still 0 because the input data has not reached there yet. These
            0 valued activations are NOT included in the return value of this
            function (so the first time steps will include less activation
            vectors than the later time steps, because the high layers haven't
            been activated yet).
        """
        self._set_activations(minibatch)

        steps = [self.activations]
        for time in xrange(1, len(self.aes) + walkback + 1):
            self._update()

            # slicing makes a shallow copy
            steps.append(self.activations[:(2 * time) + 1])

        return steps


    def get_samples(self, minibatch, walkback=0):
        """
        Runs minibatch through GSN and returns reconstructed data.

        Parameters
        ----------
        minibatch : tensor_like
            Theano symbolic representing the input minibatch.
        walkback : int
            How many walkback steps to perform. This is both how many extra
            samples to take as well as how many extra reconstructed points
            to train off of.

        Returns
        ---------
        reconstructions : list of tensor_likes
            A list of length 1 + walkback that contains the samples generated
            by the GSN. The samples will be of the same size as the minibatch.
        """
        # FIXME: should this return all of the reconstructions, or only the ones
        # that made it all the way to last layer (ie just 1 if no walkback)

        results = self._run(minibatch, walkback=walkback)
        activations = results[len(self.aes):]
        return [act[0] for act in activations]

    def __call__(self, minibatch):
        """
        As specified by StackedBlocks, this returns the output representation of
        all layers. This occurs at the final time step.
        """
        return self._run(minibatch)[-1]

    def _set_activations(self, minibatch):
        """
        Sets the input layer to minibatch and all other layers to 0.

        Parameters:
        ------------
        minibatch : tensor_like
            Theano symbolic representing the input minibatch
        """
        self.activations = [minibatch]

        for i in xrange(len(self.aes)):
            self.activations.append(
                T.zeros_like(
                    T.dot(self.activations[i], self.aes[i].weights)
                )
            )


    def _update(self):
        """
        See Figure 1 in "Deep Generative Stochastic Networks as Generative
        Models" by Bengio, Thibodeau-Laufer.
        This and _update_activations implement exactly that, which is essentially
        forward propogating the neural network in both directions.
        """

        # odd layers
        self._update_activations(xrange(1, len(self.activations), 2))

        # even even layers
        self._update_activations(xrange(0, len(self.activations), 2))

    def _update_activations(self, idx_iter):
        # FIXME: adding in this bias was breaking things. Should be adding it.
        from_above = lambda i: (self.aes[i].visbias +
                                T.dot(self.activations[i + 1],
                                      self.aes[i].weights.T))

        ''' equivalent code:
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
                pass

            # then activate!
            # pre activation corruption
            self.activations[i] = self.preact_cors[i](self.activations[i])

            # using activation function from lower autoencoder
            # no activation function on input layer
            if i != 0:
                if self.aes[i - 1].act_enc is not None:
                    self.activations[i] = self.aes[i - 1].act_enc(self.activations[i])

                # post activation corruption
                self.activations[i] = self.postact_cors[i](self.activations[i])
