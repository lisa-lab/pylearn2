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

import copy
import functools
import itertools
import warnings

import numpy as np
import theano
T = theano.tensor

import pylearn2
from pylearn2.base import StackedBlocks
from pylearn2.corruption import BinomialSampler, MultinomialSampler
from pylearn2.models.autoencoder import Autoencoder
from pylearn2.models.model import Model
from pylearn2.utils import safe_zip

def plushmax(x, eps=0.0, min_val=0.0):
    """
    A softer softmax.

    Instead of computing exp(x_i) / sum_j(exp(x_j)), this computes
    (exp(x_i) + eps) / sum_j(exp(x_j) + eps)

    Additionally, all values in the return vector will be at least min_val.
    eps may be increased to satisfy this constraint.
    """
    assert eps >= 0.0

    MIN_SAFE = max(0.00001, min_val)

    s = T.sum(T.exp(x), axis=1, keepdims=True)
    safe_eps = (MIN_SAFE * s) / (1.0 - x.shape[1] * MIN_SAFE)
    safe_eps = T.cast(safe_eps, theano.config.floatX)

    eps = T.maximum(eps, safe_eps)

    y = x + T.log(1.0 + eps * T.exp(-x))
    return T.nnet.softmax(y)

class GSN(StackedBlocks, Model):
    """
    Note: Anyone trying to use this class should read the docstring for
    GSN._set_activations. This describes the format for the minibatch parameter
    that many methods require.
    """
    def __init__(self, autoencoders, preact_cors=None, postact_cors=None,
                 layer_samplers=None):
        """
        Initialize an Generative Stochastic Network (GSN) object.

        Parameters
        ----------
        autoencoders : list
            A list of autoencoder objects. As of now, only the functionality
            from the base Autoencoder class is used.
        preact_cors : list
            A list of length len(autoencoders) + 1 where each element is a
            callable (which includes Corruptor objects). The callable at index
            i is called before activating the ith layer. Name stands for
            "preactivation corruptors".
            Valid values: None or list of length len(autoencoders) + 1. If the
            list contains a non-callable corruptor for a layer, then no
            corruption function is applied for that layer.
        postact_cors : list
            A list of length len(autoencoders) + 1 where each element is a
            callable (which includes Corruptor objects). The callable at index
            i is called directly after activating the ith layer. Name stands for
            "postactivation corruptors".
            The valid values for this parameter are the same as that for
            preact_cors.
        layer_samplers: list
            Describes how to sample from each layer. Sampling occurs directly
            before the post activation corruption is applied.
            Valid values for this argument are of the same form as valid parameters
            for preact_cor and postact_cor (and if an element in the list is None,
            no sampling will be applied at that layer). Note: as of right now,
            we've only experimented with sampling at the visible layer.

        Notes
        -----
        Most of the time it will be much easier to construct a GSN using GSN.new
        rather than GSN.__init__. This method exists to make the GSN class very
        easy to modify.
        The activation function for the visible layer is the "act_dec"
        function on the first autoencoder, and the activation function for the ith
        hidden layer is the "act_enc" function on the (i - 1)th autoencoder.
        """
        super(GSN, self).__init__(autoencoders)

        # only for convenience
        self.aes = self._layers

        # easy way to turn off corruption (True => corrupt, False => don't)
        self._corrupt_switch = True

        # easy way to not use bias (True => use bias, False => don't)
        self._bias_switch = True

        for i, ae in enumerate(self.aes):
            if i != 0:
                if ae.weights is None:
                    ae.set_visible_size(self.aes[i - 1].nhid)
                else:
                    assert (ae.weights.get_value().shape[0] ==
                            self.aes[i - 1].nhid)

        def _make_callable_list(previous):
            num_activations = len(self.aes) + 1
            identity = pylearn2.utils.identity
            if previous is None:
                previous = [identity] * num_activations

            assert len(previous) == num_activations

            for i, f in enumerate(previous):
                if not callable(f):
                    previous[i] = identity
            return previous

        self._preact_cors = _make_callable_list(preact_cors)
        self._postact_cors = _make_callable_list(postact_cors)
        self._layer_samplers = _make_callable_list(layer_samplers)

    @staticmethod
    def _make_aes(layer_sizes, activation_funcs, tied=True):
        """
        Creates the Autoencoder objects needed by the GSN.
        """
        aes = []
        assert len(activation_funcs) == len(layer_sizes)

        for i in xrange(len(layer_sizes) - 1):
            # activation for visible layer is aes[0].act_dec
            act_enc = activation_funcs[i + 1]
            act_dec = act_enc if i != 0 else activation_funcs[0]
            aes.append(
                Autoencoder(layer_sizes[i], layer_sizes[i + 1],
                            act_enc, act_dec, tied_weights=tied)
            )

        return aes

    @classmethod
    def new(cls,
            layer_sizes,
            activation_funcs,
            pre_corruptors,
            post_corruptors,
            layer_samplers,
            tied=True):

        aes = cls._make_aes(layer_sizes, activation_funcs, tied=tied)

        return cls(aes,
                   preact_cors=pre_corruptors,
                   postact_cors=post_corruptors,
                   layer_samplers=layer_samplers)

    @classmethod
    def new_ae(cls, layer_sizes, vis_corruptor=None, hidden_pre_corruptor=None,
               hidden_post_corruptor=None, visible_act="sigmoid",
               visible_sampler=BinomialSampler(),
               hidden_act="tanh", tied=True, only_corrupt_top=True):
        """
        This is just a convenience method to initialize GSN instances. The
        __init__ method is far more general, but this should capture most
        of the GSN use cases.

        Note
        ----
        This is used only for the autoencoder like GSNs described in the papers
        referenced above. GSNs for supervised learning should not use this
        method.

        Parameters
        ----------
        layer_sizes : list of integers
            Each element of this list states the size of a layer in the GSN. The
            first element in this list is the size of the visual layer.
        vis_corruptor : callable
            A callable object (such as a Corruptor) that is used to corrupt the
            visible layer.
        hidden_pre_corruptor : callable
            Same sort of object as the vis_corruptor, used to corrupt (add noise)
            the top hidden activations prior to activation.
        hidden_post_corruptor : callable
            Same sort of object as the other corruptors, used to corrupt top
            hidden activation after activation.
        visible_act : callable or string
            The value for visible_act must be a valid value for the act_dec
            parameter of Autoencoder.__init__
        hidden_act : callable or string
            The value for visible_act must be a valid value for the act_enc
            parameter of Autoencoder.__init__
        """
        num_hidden = len(layer_sizes) - 1
        activations = [visible_act] + [hidden_act] * num_hidden

        if not only_corrupt_top:
            pre_corruptors = [None] + [hidden_pre_corruptor] * num_hidden
            post_corruptors = [vis_corruptor] + [hidden_post_corruptor] * num_hidden
        else:
            pre_corruptors = [None] * num_hidden + [hidden_pre_corruptor]
            post_corruptors = [vis_corruptor] + [None] * (num_hidden - 1) +\
                [hidden_post_corruptor]

        # binomial sampling on visible layer by default
        layer_samplers = [visible_sampler] + [None] * num_hidden

        return cls.new(layer_sizes, activations, pre_corruptors, post_corruptors,
                       layer_samplers, tied=tied)

    @classmethod
    def new_classifier(cls, layer_sizes, vis1_pre_corruptor=None,
                       vis1_post_corruptor=None, vis2_pre_corruptor=None,
                       vis2_post_corruptor=None,
                       hidden_pre_corruptor=None, hidden_post_corruptor=None,
                       visible_act="sigmoid", hidden_act="tanh",
                       classifier_act=plushmax, tied=True):
        """
        FIXME: documentation
        """
        num_hidden = len(layer_sizes) - 2
        activations = [visible_act] + [hidden_act] * (num_hidden) + [classifier_act]

        pre_corruptors = [vis1_pre_corruptor] + [hidden_pre_corruptor] * num_hidden +\
            [vis2_pre_corruptor]
        post_corruptors = [vis1_post_corruptor] + [hidden_post_corruptor] * num_hidden +\
            [vis2_post_corruptor]

        layer_samplers = [BinomialSampler()] + [None] * num_hidden +\
            [MultinomialSampler()]

        return cls.new(layer_sizes, activations, pre_corruptors, post_corruptors,
                       layer_samplers, tied=tied)

    @functools.wraps(Model.get_params)
    def get_params(self):
        params = []
        for ae in self.aes:
            params.extend(ae.get_params())
        return params

    def _run(self, minibatch, walkback=0, clamped=None):
        """
        This runs the GSN on input 'minibatch' are returns all of the activations
        at every time step.

        Parameters
        ----------
        minibatch : see parameter description in _set_activations
        walkback : int
            How many walkback steps to perform.
        clamped : list of theano tensors or None.
            clamped must be None or a list of len(minibatch) where each element
            is a Theano tensor or None. Each Theano tensor should be 1 for
            indices where the value should be clamped and 0 for where the value
            should not be clamped.

        Returns
        ---------
        steps : list of list of tensor_likes
            A list of the activations at each time step. The activations
            themselves are lists of tensor_like symbolic (shared) variables.
            A time step consists of a call to the _update function (so updating
            both the odd and even layers).
        """
        # the indices which are being set
        set_idxs = safe_zip(*minibatch)[0]

        diff = lambda L: [L[i] - L[i - 1] for i in xrange(1, len(L))]
        assert 1 not in diff(sorted(set_idxs)), "Cannot set adjacent layers"

        self._set_activations(minibatch)

        # intialize steps
        steps = [self.activations[:]]

        self.apply_postact_corruption(self.activations,
                                      xrange(len(self.activations)))

        if clamped is not None:
            vals = safe_zip(*minibatch)[1]
            clamped = safe_zip(set_idxs, vals, clamped)

        # main loop
        for _ in xrange(len(self.aes) + walkback):
            steps.append(self._update(self.activations, clamped=clamped))

        return steps

    def _make_or_get_compiled(self, indices, clamped=False):
        def compile_f_init():
            mb = T.matrices(len(indices))
            zipped = safe_zip(indices, mb)
            f_init = theano.function(mb,
                                     self._set_activations(zipped, corrupt=True))
            # handle splitting of concatenated data
            def wrap_f_init(*args):
                data = f_init(*args)
                length = len(data) / 2
                return data[:length], data[length:]
            return wrap_f_init

        def compile_f_step():
            prev = T.matrices(len(self.activations))
            if clamped:
                _initial = T.matrices(len(indices))
                _clamps = T.matrices(len(indices))

                z = self._update(copy.copy(prev),
                                 clamped=safe_zip(indices, _initial, _clamps),
                                 return_activations=True)
                f = theano.function(prev + _initial + _clamps, z,
                                    on_unused_input='ignore',
                                    allow_input_downcast=True)
            else:
                z = self._update(copy.copy(prev), return_activations=True)
                f = theano.function(prev, z, on_unused_input='ignore')

            def wrapped(*args):
                data = f(*args)
                length = len(data) / 2
                return data[:length], data[length:]

            return wrapped

        if hasattr(self, '_compiled_cache'):
            if indices == self._compiled_cache[0]:
                return self._compiled_cache[1:]
            else:
                f_init = compile_f_init()
                cc = self._compiled_cache
                self._compiled_cache = (indices, f_init, cc[2])
                return self._compiled_cache[1:]

        f_init = compile_f_init()
        f_step = compile_f_step()

        self._compiled_cache = (indices, f_init, f_step)
        return self._compiled_cache[1:]

    def get_samples(self, minibatch, walkback=0, indices=None, symbolic=True,
                    include_first=False, clamped=None):
        """
        Runs minibatch through GSN and returns reconstructed data.

        This function

        Parameters
        ----------
        minibatch : see parameter description in _set_activations
            In addition to the description in get_samples, the tensor_likes
            in the list should be replaced by numpy matrices if symbolic=False.
        walkback : int
            How many walkback steps to perform. This is both how many extra
            samples to take as well as how many extra reconstructed points
            to train off of.
        indices : None or list of ints
            Indices of the layers that should be returned for each time step.
            If indices is None, then get_samples returns the values for all of
            the layers which were initially specified (by minibatch).
        symbolic : bool
            Whether the input (minibatch) contains a sparse array of Theano
            (symbolic) tensors or actual (numpy) arrays. This flag is needed
            because Theano cannot compile the large computational graphs that
            walkback creates.
        include_first : bool
            Whether to include the initial activations (ie just the input) in
            the output. This is useful for visualization, but can screw up
            training due to some cost functions failing on perfect reconstruction.
        clamped : list of tensor_likes
            See description on _run. Theano symbolics should be replaced by
            numpy matrices if symbolic=False.
            Length must be the same as length of minibatch.

        Returns
        ---------
        reconstructions : list of tensor_likes
            A list of length 1 + walkback that contains the samples generated
            by the GSN. The samples will be of the same size as the minibatch.
        """
        if walkback > 8 and symbolic:
            warnings.warn(("Running GSN in symbolic mode (needed for training) " +
                           "with a lot of walkback. Theano may take a very long " +
                           "time to compile this computational graph. If " +
                           "compiling is taking too long, then reduce the amount " +
                           "of walkback."))

        input_idxs = safe_zip(*minibatch)[0]
        if indices is None:
            indices = input_idxs

        if not symbolic:
            vals = safe_zip(*minibatch)[1]
            f_init, f_step = self._make_or_get_compiled(input_idxs,
                                                        clamped=clamped is not None)

            if clamped is None:
                get_args = lambda x: x
            else:
                mb_values = [mb[1] for mb in minibatch]
                get_args = lambda x: x + mb_values + clamped

            precor, activations = f_init(*vals)
            results = [precor]
            for _ in xrange(len(self.aes) + walkback):
                precor, activations = f_step(*get_args(activations))
                results.append(precor)
        else:
            results = self._run(minibatch, walkback=walkback, clamped=clamped)

        # leave out the first time step
        if not include_first:
            results = results[1:]

        return [[step[i] for i in indices] for step in results]

    @functools.wraps(Autoencoder.reconstruct)
    def reconstruct(self, minibatch):
        # included for compatibility with cost functions for autoencoders,
        # so assumes model is in unsupervised mode
        return self.get_samples(minibatch, walkback=0, indices=[0])

    def __call__(self, minibatch):
        """
        As specified by StackedBlocks, this returns the output representation of
        all layers. This occurs at the final time step.
        """
        return self._run(minibatch)[-1]

    """
    NOTE: The following methods contain the algorithmic content of the GSN class.
    All of these methods are written in a way such that they can be run without
    modifying the state of the GSN object. This primary visible consequence of this
    is that the methods take an "activations parameter", which is generally just
    self.activations.
    Although this style is a bit odd, it is completely necessary. Theano can handle
    small amounts of walkback (which allows us to train for walkback), but for
    many sampling iterations (ie more than 10) Theano takes an extremely long time
    to compile these large computational graphs. Making all of these methods
    take activations as an explicit parameter (which they then modify in place,
    which allows calling with self.activations) allows one to create smaller
    external Theano functions that allow many sampling iterations.
    See pylearn2.models.tests.test_gsn.sampling_test for an example.
    """

    def _set_activations(self, minibatch, set_val=True, corrupt=False):
        """
        Sets the input layer to minibatch and all other layers to 0.

        Parameters:
        ------------
        minibatch : tensor_like or list of (int, tensor_like)
            Theano symbolic representing the input minibatch. See
            description for sparse parameter.
            The minibatch
            parameter must be a list of tuples of form (int, tensor_like),
            where the int component represents the index of the layer
            (so 0 for visible, -1 for top/last layer) and the tensor_like
            represents the activation at that level. Components not included
            in the sparse activations will be set to 0. For tuples included
            in the sparse activation, the tensor_like component can actually
            be None; this will result in that activation getting set to 0.
        set_val : bool
            Determines whether the method sets self.activations.
        Note
        ----
        This method creates a new list, not modifying an existing list.
        """
        activations = [None] * (len(self.aes) + 1)

        mb_size = minibatch[0][1].shape[0]
        first_layer_size = self.aes[0].weights.shape[0]

        # zero out activations to start
        activations[0] = T.alloc(0, mb_size, first_layer_size)
        for i in xrange(1, len(activations)):
            activations[i] = T.zeros_like(
                T.dot(activations[i - 1], self.aes[i - 1].weights)
            )

        # set minibatch
        indices = [t[0] for t in minibatch if t[1] is not None]
        for i, val in minibatch:
            activations[i] = val

        self._update_odds(activations, skip_idxs=indices, corrupt=False)

        if set_val:
            self.activations = activations

        if corrupt:
            return (activations +
                    self.apply_postact_corruption(activations[:],
                                                  xrange(len(activations))))
        else:
            return activations

    def _update_odds(self, activations, skip_idxs=frozenset(), corrupt=True,
                     clamped=None):
        # Update and corrupt all of the odd layers (which we aren't skipping)
        odds = filter(lambda i: i not in skip_idxs,
                      range(1, len(activations), 2))

        self._update_activations(activations, odds)

        if clamped is not None:
            self._apply_clamping(activations, clamped)

        odds_copy = [(i, activations[i]) for i in xrange(1, len(activations), 2)]

        if corrupt:
            self.apply_postact_corruption(activations, odds)

        return odds_copy

    def _update_evens(self, activations, clamped=None):
        evens = xrange(0, len(activations), 2)

        self._update_activations(activations, evens)
        if clamped is not None:
            self._apply_clamping(activations, clamped)

        evens_copy = [(i, activations[i]) for i in evens]
        self.apply_postact_corruption(activations, evens)

        return evens_copy

    def _update(self, activations, clamped=None, return_activations=False):
        """
        See Figure 1 in "Deep Generative Stochastic Networks as Generative
        Models" by Bengio, Thibodeau-Laufer.
        This and _update_activations implement exactly that, which is essentially
        forward propogating the neural network in both directions.

        Parameters
        ----------
        activations : list of tensors
            List of activations at time step t - 1.

        Returns
        -------
        y : list of tensors
            List of activations at time step t (prior to adding postact noise).

        Note
        ----
        The return value is generally not equal to the value of activations at the
        the end of this method. The return value contains all layers without
        postactivation noise, but the activations value contains noise on the
        odd layers (necessary to compute the even layers).
        """
        evens_copy = self._update_evens(activations, clamped=clamped)
        odds_copy = self._update_odds(activations, clamped=clamped)

        # precor is before sampling + postactivation corruption (after preactivation
        # corruption and activation)
        precor = [None] * len(self.activations)
        for idx, val in evens_copy + odds_copy:
            assert precor[idx] is None
            precor[idx] = val
        assert None not in precor

        if return_activations:
            return precor + activations
        else:
            return precor

    @staticmethod
    def _apply_clamping(activations, clamped, symbolic=True):
        for idx, initial, clamp in clamped:
            if clamp is None:
                continue

            # take values from initial
            clamped_val = clamp * initial

            # zero out values in activations
            if symbolic:
                zerod = activations[idx] * T.eq(clamp, 0.0)
            else:
                zerod = activations[idx] * (clamp == 0.0)

            activations[idx] = zerod + clamped_val
        return activations

    @staticmethod
    def _apply_corruption(activations, corruptors, idx_iter):
        """
        Applies post activation corruption to layers.

        Parameters
        ----------
        activations : list of tensor_likes
            Generally gsn.activations
        corruptors : list of callables
            Generally gsn.postact_cors or gsn.preact_cors
        idx_iter : iterable
            An iterable of indices into self.activations. The indexes indicate
            which layers the post activation corruptors should be applied to.
        """
        assert len(corruptors) == len(activations)
        for i in idx_iter:
            activations[i] = corruptors[i](activations[i])
        return activations

    def apply_preact_corruption(self, activations, idx_iter):
        if self._corrupt_switch:
            self._apply_corruption(activations, self._preact_cors,
                                   idx_iter)
        return activations

    def apply_postact_corruption(self, activations, idx_iter, sample=True):
        if sample:
            self.apply_sampling(activations, idx_iter)
        if self._corrupt_switch:
            self._apply_corruption(activations, self._postact_cors,
                                   idx_iter)
        return activations

    def apply_sampling(self, activations, idx_iter):
        # using _apply_corruption to apply samplers
        return self._apply_corruption(activations, self._layer_samplers,
                                      idx_iter)

    def _update_activations(self, activations, idx_iter):
        """
        Parameters
        ----------
        activations : list of tensor_likes
            The activations to update (could be self.activations). Updates
            in-place.
        idx_iter : iterable
            An iterable of indices into self.activations. The indexes indicate
            which layers should be updated.
            Must be able to iterate over idx_iter multiple times.
        """
        from_above = lambda i: ((self.aes[i].visbias if self._bias_switch else 0) +
                                T.dot(activations[i + 1],
                                      self.aes[i].w_prime))

        from_below = lambda i: ((self.aes[i - 1].hidbias if self._bias_switch else 0) +
                                T.dot(activations[i - 1],
                                     self.aes[i - 1].weights))

        for i in idx_iter:
            # first compute the hidden activation
            if i == 0:
                activations[i] = from_above(i)
            elif i == len(activations) - 1:
                activations[i] = from_below(i)
            else:
                activations[i] = from_below(i) + from_above(i)

        self.apply_preact_corruption(activations, idx_iter)

        for i in idx_iter:
            # Using the activation function from lower autoencoder
            act_func = None
            if i == 0:
                act_func = self.aes[0].act_dec
            else:
                act_func = self.aes[i - 1].act_enc

            # ACTIVATION
            # None implies linear
            if act_func is not None:
                activations[i] = act_func(activations[i])

class JointGSN(GSN):
    """
    This class only provides a few convenient methods on top of the GSN class
    above. This class should be used when learning the joint distribution between
    2 or more variables.
    """
    @classmethod
    def convert(cls, gsn, input_idx, label_idx):
        gsn.__class__ = cls
        gsn.input_idx = input_idx
        gsn.label_idx = label_idx
        return gsn

    def calc_walkback(self, trials):
        wb = trials - len(self.aes)
        if wb <= 0:
            return 0
        else:
            return wb

    def classify(self, inputs, trials=10, skip=0):
        """
        Parameters
        ----------
        FIXME: write me
        """
        clamped = np.ones(inputs.shape, dtype=np.float32)

        data = self.get_samples([(self.input_idx, inputs)],
                                walkback=self.calc_walkback(trials + skip),
                                indices=[self.label_idx],
                                clamped=[clamped],
                                symbolic=False)

        # 3d tensor: axis 0 is time step, axis 1 is minibatch item,
        # axis 2 is softmax output for label
        data = np.array(list(itertools.chain(*data[skip:skip+trials])))

        mean = data.mean(axis=0)
        am = np.argmax(mean, axis=1)

        # convert argmax's to one-hot format
        labels = np.zeros_like(mean)
        labels[np.arange(labels.shape[0]), am] = 1.0

        return labels

    def get_samples_from_labels(self, labels, trials=5):
        clamped = np.ones(labels.shape, dtype=np.float32)
        data = self.get_samples([(self.label_idx, labels)],
                                walkback=self.calc_walkback(trials),
                                indices=[self.input_idx],
                                clamped=[clamped],
                                symbolic=False)

        return np.array(list(itertools.chain(*data)))
