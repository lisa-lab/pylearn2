"""
Generative Stochastic Networks

This is described in:

- "Generalized Denoising Auto-Encoders as Generative Models" Bengio, Yao, Alain,
   Vincent. arXiv:1305.6663
- "Deep Generative Stochastic Networks Trainable by Backprop" Bengio,
   Thibodeau-Laufer. arXiv:1306.1091

There is an example of training both unsupervised and supervised GSNs on MNIST
in pylearn2/scripts/gsn_example.py
"""
__authors__ = "Eric Martin"
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"

import copy
import functools
import warnings

import numpy as np
from theano.compat.six.moves import xrange
import theano
T = theano.tensor

from pylearn2.blocks import StackedBlocks
from pylearn2.expr.activations import identity
from pylearn2.models.autoencoder import Autoencoder
from pylearn2.models.model import Model
from pylearn2.utils import safe_zip

# Enforce correct restructured text list format.
# Be sure to re-run docgen.py and make sure there are no warnings if you
# modify the module-level docstring.
assert """:

- """ in __doc__

class GSN(StackedBlocks, Model):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    autoencoders : list
        A list of autoencoder objects. As of now, only the functionality
        from the base Autoencoder class is used.
    preact_cors : list
        A list of length len(autoencoders) + 1 where each element is a
        callable (which includes Corruptor objects). The callable at
        index i is called before activating the ith layer. Name stands
        for "preactivation corruptors".
    postact_cors : list
        A list of length len(autoencoders) + 1 where each element is a
        callable (which includes Corruptor objects). The callable at
        index i is called directly after activating the ith layer. Name
        stands for "postactivation corruptors". The valid values for this
        parameter are the same as that for preact_cors.
    layer_samplers: list
        Describes how to sample from each layer. Sampling occurs directly
        before the post activation corruption is applied. Valid values
        for this argument are of the same form as valid parameters for
        preact_cor and postact_cor (and if an element in the list is
        None, no sampling will be applied at that layer). Note: as of
        right now, we've only experimented with sampling at the visible
        layer.

    Notes
    -----
    Most of the time it will be much easier to construct a GSN using
    GSN.new rather than GSN.__init__. This method exists to make the GSN
    class very easy to modify.

    The activation function for the visible layer is the "act_dec" function
    on the first autoencoder, and the activation function for the i_th
    hidden layer is the "act_enc" function on the (i - 1)th autoencoder.
    """

    def __init__(self, autoencoders, preact_cors=None, postact_cors=None,
                 layer_samplers=None):
        super(GSN, self).__init__(autoencoders)

        # only for convenience
        self.aes = self._layers

        # easy way to turn off corruption (True => corrupt, False => don't)
        self._corrupt_switch = True

        # easy way to turn off sampling
        self._sample_switch = True

        # easy way to not use bias (True => use bias, False => don't)
        self._bias_switch = True

        # check that autoencoders are the correct sizes by looking at previous
        # layer. We can't do this for the first ae, so we skip it.
        for i in xrange(1, len(self.aes)):
            assert (self.aes[i].weights.get_value().shape[0] ==
                    self.aes[i - 1].nhid)


        # do some type checking and convert None's to identity function
        def _make_callable_list(previous):
            """
            .. todo::

                WRITEME
            """
            if len(previous) != self.nlayers:
                raise ValueError("Need same number of corruptors/samplers as layers")

            if not all(map(lambda x: callable(x) or x is None, previous)):
                raise ValueError("All elements must either be None or be a callable")

            return map(lambda x: identity if x is None else x, previous)

        self._preact_cors = _make_callable_list(preact_cors)
        self._postact_cors = _make_callable_list(postact_cors)
        self._layer_samplers = _make_callable_list(layer_samplers)

    @staticmethod
    def _make_aes(layer_sizes, activation_funcs, tied=True):
        """
        Creates the Autoencoder objects needed by the GSN.

        Parameters
        ----------
        layer_sizes : WRITEME
        activation_funcs : WRITEME
        tied : WRITEME
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
        """
        An easy (and recommended) way to initialize a GSN.

        Parameters
        ----------
        layer_sizes : list
            A list of integers. The i_th element in the list is the size of
            the i_th layer of the network, and the network will have
            len(layer_sizes) layers.
        activation_funcs : list
            activation_funcs must be a list of the same length as layer_sizes
            where the i_th element is the activation function for the i_th
            layer. Each component of the list must refer to an activation
            function in such a way that the Autoencoder class recognizes the
            function. Valid values include a callable (which takes a symbolic
            tensor), a string that refers to a Theano activation function, or
            None (which gives the identity function).
        preact_corruptors : list
            preact_corruptors follows exactly the same format as the
            activations_func argument.
        postact_corruptors : list
            postact_corruptors follows exactly the same format as the
            activations_func argument.
        layer_samplers : list
            layer_samplers follows exactly the same format as the
            activations_func argument.
        tied : bool
            Indicates whether the network should use tied weights.

        Notes
        -----
        The GSN classes applies functions in the following order:
          - pre-activation corruption
          - activation
          - clamping applied
          - sampling
          - post-activation corruption

        All setting and returning of values occurs after applying the
        activation function (or clamping if clamping is used) but before
        applying sampling.
        """
        args = [layer_sizes, pre_corruptors, post_corruptors, layer_samplers]
        if not all(isinstance(arg, list) for arg in args):
            raise TypeError("All arguments except for tied must be lists")
        if not all(len(arg) == len(args[0]) for arg in args):
            lengths = map(len, args)
            raise ValueError("All list arguments must be of the same length. " +
                             "Current lengths are %s" % lengths)

        aes = cls._make_aes(layer_sizes, activation_funcs, tied=tied)

        return cls(aes,
                   preact_cors=pre_corruptors,
                   postact_cors=post_corruptors,
                   layer_samplers=layer_samplers)

    @functools.wraps(Model.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        params = set()
        for ae in self.aes:
            params.update(ae.get_params())
        return list(params)

    @property
    def nlayers(self):
        """
        Returns how many layers the GSN has.
        """
        return len(self.aes) + 1

    def _run(self, minibatch, walkback=0, clamped=None):
        """
        This runs the GSN on input 'minibatch' and returns all of the activations
        at every time step.

        Parameters
        ----------
        minibatch : see parameter description in _set_activations
        walkback : int
            How many walkback steps to perform.
        clamped : list of theano tensors or None.
            clamped must be None or a list of len(minibatch) where each
            element is a Theano tensor or None. Each Theano tensor should be
            1 for indices where the value should be clamped and 0 for where
            the value should not be clamped.

        Returns
        -------
        steps : list of list of tensor_likes
            A list of the activations at each time step. The activations
            themselves are lists of tensor_like symbolic variables.
            A time step consists of a call to the _update function
            (so updating both the even and odd layers). When there is no
            walkback, the GSN runs long enough for signal from the bottom
            layer to propogate to the top layer and then back to the bottom.
            The walkback parameter adds single steps on top of the default.
        """
        # the indices which are being set
        set_idxs = safe_zip(*minibatch)[0]

        if self.nlayers == 2 and len(set_idxs) == 2:
            if clamped is None:
                raise ValueError("Setting both layers of 2 layer GSN without " +
                                 "clamping causes one layer to overwrite the " +
                                 "other. The value for layer 0 will not be used.")
            else:
                warnings.warn("Setting both layers of 2 layer GSN with clamping " +
                              "may not be valid, depending on what clamping is " +
                              "done")

        diff = lambda L: [L[i] - L[i - 1] for i in xrange(1, len(L))]
        if 1 in diff(sorted(set_idxs)):
            # currently doing an odd step at first. If this warning appears, you
            # should remember that the odd step (ie calculating the odd activations)
            # is done first (so all setting of odd layers is valid) and that for
            # an even layer to have an effect it must be used to compute either the
            # (odd) layer below or above it.
            warnings.warn("Adjacent layers in the GSN are being set. There is a" +
                          " significant possibility that some of the set values" +
                          " are not being used and are just overwriting each " +
                          "other. This is dependent on both the ordering of the " +
                          "even and odd steps as well as the proximity to the " +
                          "edge of the network.\n It is recommended to read the " +
                          "source to ensure the behavior is understood if setting " +
                          "adjacent layers.")

        self._set_activations(minibatch)

        # intialize steps
        steps = [self.activations[:]]

        self.apply_postact_corruption(self.activations,
                                      xrange(self.nlayers))

        if clamped is not None:
            vals = safe_zip(*minibatch)[1]
            clamped = safe_zip(set_idxs, vals, clamped)

        # main loop
        for _ in xrange(len(self.aes) + walkback):
            steps.append(self._update(self.activations, clamped=clamped))

        return steps

    def _make_or_get_compiled(self, indices, clamped=False):
        """
        Compiles, wraps, and caches Theano functions for non-symbolic calls
        to get_samples.

        Parameters
        ----------
        indices : WRITEME
        clamped : WRITEME
        """
        def compile_f_init():
            mb = T.matrices(len(indices))
            zipped = safe_zip(indices, mb)
            f_init = theano.function(mb,
                                     self._set_activations(zipped, corrupt=True),
                                     allow_input_downcast=True)
            # handle splitting of concatenated data
            def wrap_f_init(*args):
                data = f_init(*args)
                length = len(data) / 2
                return data[:length], data[length:]
            return wrap_f_init

        def compile_f_step():
            prev = T.matrices(self.nlayers)
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
                f = theano.function(prev, z, on_unused_input='ignore',
                                    allow_input_downcast=True)

            def wrapped(*args):
                data = f(*args)
                length = len(data) / 2
                return data[:length], data[length:]

            return wrapped

        # things that require re-compiling everything
        state = (self._corrupt_switch, self._sample_switch, self._bias_switch)

        if hasattr(self, '_compiled_cache') and state == self._compiled_cache[0]:
            # already have some cached functions

            if indices == self._compiled_cache[1]:
                # everything is cached, return all but state and indices
                return self._compiled_cache[2:]
            else:
                # indices have changed, need to recompile f_init
                f_init = compile_f_init()
                cc = self._compiled_cache
                self._compiled_cache = (state, indices, f_init, cc[3])
                return self._compiled_cache[2:]
        else:
            # have no cached function (or incorrect state)
            f_init = compile_f_init()
            f_step = compile_f_step()
            self._compiled_cache = (state, indices, f_init, f_step)
            return self._compiled_cache[2:]

    def get_samples(self, minibatch, walkback=0, indices=None, symbolic=True,
                    include_first=False, clamped=None):
        """
        Runs minibatch through GSN and returns reconstructed data.

        Parameters
        ----------
        minibatch : see parameter description in _set_activations
            In addition to the description in get_samples, the tensor_likes
            in the list should be replaced by numpy matrices if symbolic=False.
        walkback : int
            How many walkback steps to perform. This is both how many extra
            samples to take as well as how many extra reconstructed points
            to train off of. See description in _run.
            This parameter controls how many samples you get back.
        indices : None or list of ints, optional
            Indices of the layers that should be returned for each time step.
            If indices is None, then get_samples returns the values for all
            of the layers which were initially specified (by minibatch).
        symbolic : bool, optional
            Whether the input (minibatch) contains a Theano (symbolic)
            tensors or actual (numpy) arrays. This flag is needed because
            Theano cannot compile the large computational graphs that
            walkback creates.
        include_first : bool, optional
            Whether to include the initial activations (ie just the input) in
            the output. This is useful for visualization, but can screw up
            training due to some cost functions failing on perfect
            reconstruction.
        clamped : list of tensor_likes, optional
            See description on _run. Theano symbolics should be replaced by
            numpy matrices if symbolic=False. Length must be the same as
            length of minibatch.

        Returns
        -------
        reconstructions : list of tensor_likes
            A list of length 1 + number of layers + walkback that contains
            the samples generated by the GSN. The layers returned at each
            time step is decided by the indices parameter (and defaults to
            the layers specified in minibatch). If include_first is True,
            then the list will be 1 element longer (inserted at beginning)
            than specified above.
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
        """
        .. todo::

            WRITEME
        """
        # included for compatibility with cost functions for autoencoders,
        # so assumes model is in unsupervised mode

        assert len(minibatch) == 1
        idx = minibatch[0][0]
        return self.get_samples(minibatch, walkback=0, indices=[idx])

    def __call__(self, minibatch):
        """
        As specified by StackedBlocks, this returns the output representation of
        all layers. This occurs at the final time step.

        Parameters
        ----------
        minibatch : WRITEME

        Returns
        -------
        WRITEME
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
    many sampling iterations (ie more than 10) Theano struggles to compile these
    large computational graphs. Making all of these methods below take
    activations as an explicit parameter (which they then modify in place,
    which allows calling with self.activations) allows one to create smaller
    external Theano functions that allow many sampling iterations.
    See pylearn2.models.tests.test_gsn.sampling_test for an example.
    """

    def _set_activations(self, minibatch, set_val=True, corrupt=False):
        """
        Initializes the GSN as specified by minibatch.

        Parameters
        ----------
        minibatch : list of (int, tensor_like)
            The minibatch parameter must be a list of tuples of form
            (int, tensor_like), where the int component represents the index
            of the layer (so 0 for visible, -1 for top/last layer) and the
            tensor_like represents the activation at that level. Layer
            indices not included in the minibatch will be set to 0. For
            tuples included in the minibatch, the tensor_like component can
            actually be None; this will result in that layer getting set to 0
            initially.
        set_val : bool, optional
            Determines whether the method sets self.activations.
        corrupt : bool, optional
            Instructs the method to return both a non-corrupted and corrupted
            set of activations rather than just non-corrupted.

        Notes
        -----
        This method creates a new list, not modifying an existing list.
        This method also does the first odd step in the network.
        """
        activations = [None] * self.nlayers

        mb_size = minibatch[0][1].shape[0]
        first_layer_size = self.aes[0].weights.shape[0]

        # zero out activations to start
        activations[0] = T.alloc(0, mb_size, first_layer_size)
        for i in xrange(1, len(activations)):
            activations[i] = T.zeros_like(
                T.dot(activations[i - 1], self.aes[i - 1].weights)
            )

        # set minibatch
        for i, val in minibatch:
            if val is not None:
                activations[i] = val

        indices = [t[0] for t in minibatch if t[1] is not None]
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
        """
        Updates just the odd layers of the network.

        Parameters
        ----------
        activations : list
            List of symbolic tensors representing the current activations.
        skip_idxs : list
            List of integers representing which odd indices should not be
            updated. This parameter exists so that _set_activations can solve
            the tricky problem of initializing the network when both even and
            odd layers are being assigned.
        corrupt : bool, optional
            Whether or not to apply post-activation corruption to the odd
            layers. This parameter does not alter the return value of this
            method but does modify the activations parameter in place.
        clamped : list, optional
            See description for _apply_clamping.
        """
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
        """
        Updates just the even layers of the network.

        Parameters
        ----------
        See all of the descriptions for _update_evens.
        """
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
        clamped : list
            See description on _apply_clamping
        return_activations : bool
            If true, then this method returns both the activation values
            after the activation function has been applied and the values
            after the sampling + post-activation corruption has been applied.
            If false, then only return the values after the activation
            function has been applied (no corrupted version).
            This parameter is only set to True when compiling the functions
            needed by get_samples. Regardless of this parameter setting, the
            sampling/post-activation corruption noise is still added in-place
            to activations.

        Returns
        -------
        y : list of tensors
            List of activations at time step t (prior to adding postact noise).

        Notes
        -----
        The return value is generally not equal to the value of activations at
        the the end of this method. The return value contains all layers
        without sampling/post-activation noise, but the activations value
        contains noise on the odd layers (necessary to compute the even
        layers).
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
        """
        Resets the value of some layers within the network.

        Parameters
        ----------
        activations : list
            List of symbolic tensors representing the current activations.
        clamped : list of (int, matrix, matrix or None) tuples
            The first component of each tuple is an int representing the
            index of the layer to clamp.
            The second component is a matrix of the initial values for that
            layer (ie what we are resetting the values to).
            The third component is a matrix mask indicated which indices in
            the minibatch to clamp (1 indicates clamping, 0 indicates not).
            The value of None is equivalent to the 0 matrix (so no clamping).
            If symbolic is true then matrices are Theano tensors, otherwise
            they should be numpy matrices.
        symbolic : bool, optional
            Whether to execute with symbolic Theano tensors or numpy matrices.
        """
        for idx, initial, clamp in clamped:
            if clamp is None:
                continue

            # take values from initial
            clamped_val = clamp * initial

            # zero out values in activations
            if symbolic:
                activations[idx] = T.switch(clamp, initial, activations[idx])
            else:
                activations[idx] = np.switch(clamp, initial, activations[idx])
        return activations

    @staticmethod
    def _apply_corruption(activations, corruptors, idx_iter):
        """
        Applies a list of corruptor functions to all layers.

        Parameters
        ----------
        activations : list of tensor_likes
            Generally gsn.activations
        corruptors : list of callables
            Generally gsn.postact_cors or gsn.preact_cors
        idx_iter : iterable
            An iterable of indices into self.activations. The indexes
            indicate which layers the post activation corruptors should be
            applied to.
        """
        assert len(corruptors) == len(activations)
        for i in idx_iter:
            activations[i] = corruptors[i](activations[i])
        return activations

    def apply_sampling(self, activations, idx_iter):
        """
        .. todo::

            WRITEME
        """
        # using _apply_corruption to apply samplers
        if self._sample_switch:
            self._apply_corruption(activations, self._layer_samplers,
                                   idx_iter)
        return activations

    def apply_postact_corruption(self, activations, idx_iter, sample=True):
        """
        .. todo::

            WRITEME
        """
        if sample:
            self.apply_sampling(activations, idx_iter)
        if self._corrupt_switch:
            self._apply_corruption(activations, self._postact_cors,
                                   idx_iter)
        return activations

    def apply_preact_corruption(self, activations, idx_iter):
        """
        .. todo::

            WRITEME
        """
        if self._corrupt_switch:
            self._apply_corruption(activations, self._preact_cors,
                                   idx_iter)
        return activations

    def _update_activations(self, activations, idx_iter):
        """
        Actually computes the activations for all indices in idx_iters.

        This method computes the values for a layer by computing a linear
        combination of the neighboring layers (dictated by the weight matrices),
        applying the pre-activation corruption, and then applying the layer's
        activation function.

        Parameters
        ----------
        activations : list of tensor_likes
            The activations to update (could be self.activations). Updates
            in-place.
        idx_iter : iterable
            An iterable of indices into self.activations. The indexes
            indicate which layers should be updated.
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
    2 vectors.
    """
    @classmethod
    def convert(cls, gsn, input_idx=0, label_idx=None):
        """
        'convert' essentially serves as the constructor for JointGSN.

        Parameters
        ----------
        gsn : GSN
        input_idx : int
            The index of the layer which serves as the "input" to the
            network. During classification, this layer will be given.
            Defaults to 0.
        label_idx : int
            The index of the layer which serves as the "output" of the
            network. This label is predicted during classification.
            Defaults to top layer of network.
        """
        gsn = copy.copy(gsn)
        gsn.__class__ = cls
        gsn.input_idx = input_idx
        gsn.label_idx = label_idx or (gsn.nlayers - 1)
        return gsn

    def calc_walkback(self, trials):
        """
        Utility method that calculates how much walkback is needed to get at
        at least 'trials' samples.

        Parameters
        ----------
        trials : WRITEME
        """
        wb = trials - len(self.aes)
        if wb <= 0:
            return 0
        else:
            return wb

    def _get_aggregate_classification(self, minibatch, trials=10, skip=0):
        """
        See classify method.

        Returns the prediction vector aggregated over all time steps where
        axis 0 is the minibatch item and axis 1 is the output for the label.
        """
        clamped = np.ones(minibatch.shape, dtype=np.float32)

        data = self.get_samples([(self.input_idx, minibatch)],
                                walkback=self.calc_walkback(trials + skip),
                                indices=[self.label_idx],
                                clamped=[clamped],
                                symbolic=False)

        # 3d tensor: axis 0 is time step, axis 1 is minibatch item,
        # axis 2 is softmax output for label (after slicing)
        data = np.asarray(data[skip:skip+trials])[:, 0, :, :]

        return data.mean(axis=0)

    def classify(self, minibatch, trials=10, skip=0):
        """
        Classifies a minibatch.

        This method clamps minibatch at self.input_idx and then runs the GSN.
        The first 'skip' predictions are skipped and the next 'trials'
        predictions are averaged and then arg-maxed to make a final prediction.
        The prediction vectors are the activations at self.label_idx.

        Parameters
        ----------
        minibatch : numpy matrix
            WRITEME
        trials : int
            WRITEME
        skip : int
            WRITEME

        Notes
        -----
        A fairly large 3D tensor during classification, so one should watch
        their memory use. The easiest way to limit memory consumption is to
        classify just minibatches rather than the whole test set at once.
        The large tensor is of size (skip + trials) * mb_size * num labels.

        .. warning::

            This method does not directly control whether or not
            corruption and sampling is applied during classification.
            These are decided by self._corrupt_switch and
            self._sample_switch.
        """
        mean = self._get_aggregate_classification(minibatch, trials=trials,
                                                  skip=skip)
        am = np.argmax(mean, axis=1)

        # convert argmax's to one-hot format
        labels = np.zeros_like(mean)
        labels[np.arange(labels.shape[0]), am] = 1.0

        return labels

    def get_samples_from_labels(self, labels, trials=5):
        """
        Clamps labels and generates samples.

        Parameters
        ----------
        labels : WRITEME
        trials : WRITEME
        """
        clamped = np.ones(labels.shape, dtype=np.float32)
        data = self.get_samples([(self.label_idx, labels)],
                                walkback=self.calc_walkback(trials),
                                indices=[self.input_idx],
                                clamped=[clamped],
                                symbolic=False)

        return np.array(data)[:, 0, :, :]
