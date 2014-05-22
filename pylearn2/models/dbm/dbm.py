"""
The main DBM class
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import functools
import logging
import numpy as np
import warnings

from theano.compat import OrderedDict
from theano import tensor as T, config

from pylearn2.models import Model
from pylearn2.models.dbm import flatten
from pylearn2.models.dbm.inference_procedure import WeightDoubling
from pylearn2.models.dbm.sampling_procedure import GibbsEvenOdd
from pylearn2.utils import safe_zip, safe_izip
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


class DBM(Model):
    """
    A deep Boltzmann machine.

    See "Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton
    for details.

    Parameters
    ----------
    batch_size : int
        The batch size the model should use. Some convolutional
        LinearTransforms require a compile-time hardcoded batch size,
        otherwise this would not be part of the model specification.
    visible_layer : WRITEME
        The visible layer of the DBM.
    hidden_layers : list
        The hidden layers. A list of HiddenLayer objects. The first
        layer in the list is connected to the visible layer.
    niter : int
        Number of mean field iterations for variational inference
        for the positive phase.
    sampling_procedure : WRITEME
    inference_procedure : WRITEME
    """

    def __init__(self, batch_size, visible_layer, hidden_layers, niter,
                 sampling_procedure=None, inference_procedure=None):
        self.__dict__.update(locals())
        del self.self
        assert len(hidden_layers) >= 1

        if len(hidden_layers) > 1 and niter <= 1:
            raise ValueError("with more than one hidden layer, niter needs to "
                             "be greater than 1; otherwise mean field won't "
                             "work properly.")

        self.setup_rng()
        self.layer_names = set()
        self.visible_layer.set_dbm(self)
        for layer in hidden_layers:
            assert layer.get_dbm() is None
            layer.set_dbm(self)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)
        self._update_layer_input_spaces()
        self.force_batch_size = batch_size
        self.freeze_set = set([])
        if inference_procedure is None:
            self.setup_inference_procedure()
        self.inference_procedure.set_dbm(self)
        if sampling_procedure is None:
            self.setup_sampling_procedure()
        self.sampling_procedure.set_dbm(self)

    def get_all_layers(self):
        """
        .. todo::

            WRITEME
        """
        return [self.visible_layer] + self.hidden_layers

    def energy(self, V, hidden):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        V : tensor_like
            Theano batch of visible unit observations (must be SAMPLES, not
            mean field parameters)
        hidden : list
            List, one element per hidden layer, of batches of samples (must
            be SAMPLES, not mean field parameters)

        Returns
        -------
        rval : tensor_like
            Vector containing the energy of each sample

        Notes
        -----
        Applying this function to non-sample theano variables is not guaranteed
        to give you an expected energy in general, so don't use this that way.
        """

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state=V,
                     average=False))

        # This condition could be relaxed, but current code assumes it
        assert len(self.hidden_layers) > 0

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below=self.visible_layer.upward_state(V),
            state=hidden[0], average_below=False, average=False))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            samples_below = hidden[i-1]
            layer_below = self.hidden_layers[i-1]
            samples_below = layer_below.upward_state(samples_below)
            samples = hidden[i]
            terms.append(layer.expected_energy_term(state_below=samples_below,
                         state=samples, average_below=False, average=False))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval

    def mf(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

    def expected_energy(self, V, mf_hidden):
        """
        WRITEME

        Parameters
        ----------
        V : tensor_like
            Theano batch of visible unit observations (must be SAMPLES, not
            mean field parameters: the random variables in the expectation
            are the hiddens only)
        mf_hidden : list
            List, one element per hidden layer, of batches of variational
            parameters (must be VARIATIONAL PARAMETERS, not samples. Layers
            with analytically determined variance parameters for their mean
            field parameters will use those to integrate over the variational
            distribution, so it's not generally the same thing as measuring
            the energy at a point.)

        Returns
        -------
        rval : tensor_like
            Vector containing the expected energy of each example under the
            corresponding variational distribution.
        """

        self.visible_layer.space.validate(V)
        assert isinstance(mf_hidden, (list, tuple))
        assert len(mf_hidden) == len(self.hidden_layers)

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state=V,
                     average=False))

        # This condition could be relaxed, but current code assumes it
        assert len(self.hidden_layers) > 0

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below=self.visible_layer.upward_state(V),
            average_below=False, state=mf_hidden[0], average=True))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer_below = self.hidden_layers[i-1]
            mf_below = mf_hidden[i-1]
            mf_below = layer_below.upward_state(mf_below)
            mf = mf_hidden[i]
            terms.append(layer.expected_energy_term(state_below=mf_below,
                         state=mf, average_below=True, average=True))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval

    def setup_rng(self):
        """
        .. todo::

            WRITEME
        """
        self.rng = make_np_rng(None, [2012, 10, 17], which_method="uniform")

    def setup_inference_procedure(self):
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = WeightDoubling()
            self.inference_procedure.set_dbm(self)

    def setup_sampling_procedure(self):
        """
        .. todo::

            WRITEME
        """
        if not hasattr(self, 'sampling_procedure') or \
                self.sampling_procedure is None:
            self.sampling_procedure = GibbsEvenOdd()
            self.sampling_procedure.set_dbm(self)

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.hidden_layers[-1].get_output_space()

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        visible_layer = self.visible_layer
        hidden_layers = self.hidden_layers

        self.hidden_layers[0].set_input_space(visible_layer.space)
        for i in xrange(1, len(hidden_layers)):
            hidden_layers[i].set_input_space(
                hidden_layers[i-1].get_output_space())

        for layer in self.get_all_layers():
            layer.finalize_initialization()

    def add_layers(self, layers):
        """
        Add new layers on top of the existing hidden layers

        Parameters
        ----------
        layers : WRITEME
        """

        # Patch old pickle files
        if not hasattr(self, 'rng'):
            self.setup_rng()

        hidden_layers = self.hidden_layers
        assert len(hidden_layers) > 0
        for layer in layers:
            assert layer.get_dbm() is None
            layer.set_dbm(self)
            layer.set_input_space(hidden_layers[-1].get_output_space())
            hidden_layers.append(layer)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)

    def freeze(self, parameter_set):
        """
        .. todo::

            WRITEME
        """
        # patch old pickle files
        if not hasattr(self, 'freeze_set'):
            self.freeze_set = set([])

        self.freeze_set = self.freeze_set.union(parameter_set)

    def get_params(self):
        """
        .. todo::

            WRITEME
        """

        rval = []
        for param in self.visible_layer.get_params():
            assert param.name is not None
        rval = self.visible_layer.get_params()
        for layer in self.hidden_layers:
            for param in layer.get_params():
                if param.name is None:
                    raise ValueError("All of your parameters should have "
                                     "names, but one of " + layer.layer_name +
                                     "'s doesn't")
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        # Patch pickle files that predate the freeze_set feature
        if not hasattr(self, 'freeze_set'):
            self.freeze_set = set([])

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.hidden_layers:
            layer.set_batch_size(batch_size)

        if not hasattr(self, 'inference_procedure'):
            self.setup_inference_procedure()
        self.inference_procedure.set_batch_size(batch_size)

    @functools.wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        self.visible_layer.modify_updates(updates)
        for layer in self.hidden_layers:
            layer.modify_updates(updates)

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.visible_layer.space

    def get_lr_scalers(self):
        """
        .. todo::

            WRITEME
        """
        rval = OrderedDict()

        params = self.get_params()

        for layer in self.hidden_layers + [self.visible_layer]:
            contrib = layer.get_lr_scalers()

            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)
        assert all([isinstance(val, float) for val in rval.values()])

        return rval

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
        return self.hidden_layers[0].get_weights()

    def get_weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.hidden_layers[0].get_weights_view_shape()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return self.hidden_layers[0].get_weights_format()

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        return self.hidden_layers[0].get_weights_topo()

    def make_layer_to_state(self, num_examples, rng=None):
        """
        Makes and returns a dictionary mapping layers to states.

        By states, we mean here a real assignment, not a mean field
        state. For example, for a layer containing binary random
        variables, the state will be a shared variable containing
        values in {0,1}, not [0,1]. The visible layer will be included.

        Uses a dictionary so it is easy to unambiguously index a layer
        without needing to remember rules like vis layer = 0, hiddens
        start at 1, etc.

        Parameters
        ----------
        num_examples : int
            WRITEME
        rng : WRITEME
        """

        # Make a list of all layers
        layers = [self.visible_layer] + self.hidden_layers

        if rng is None:
            rng = self.rng

        states = [layer.make_state(num_examples, rng) for layer in layers]

        zipped = safe_zip(layers, states)

        def recurse_check(layer, state):
            if isinstance(state, (list, tuple)):
                for elem in state:
                    recurse_check(layer, elem)
            else:
                val = state.get_value()
                m = val.shape[0]
                if m != num_examples:
                    raise ValueError(layer.layer_name + " gave state with " +
                                     str(m) + " examples in some component."
                                     "We requested " + str(num_examples))

        for layer, state in zipped:
            recurse_check(layer, state)

        rval = OrderedDict(zipped)

        return rval

    def make_layer_to_symbolic_state(self, num_examples, rng=None):
        """
        .. todo::

            Explain the difference with `make_layer_to_state`

        Makes and returns a dictionary mapping layers to states.

        By states, we mean here a real assignment, not a mean field
        state. For example, for a layer containing binary random
        variables, the state will be a shared variable containing
        values in {0,1}, not [0,1]. The visible layer will be included.

        Uses a dictionary so it is easy to unambiguously index a layer
        without needing to remember rules like vis layer = 0, hiddens
        start at 1, etc.

        Parameters
        ----------
        num_examples : int
            WRITEME
        rng : WRITEME
        """

        # Make a list of all layers
        layers = [self.visible_layer] + self.hidden_layers

        assert rng is not None

        states = [layer.make_symbolic_state(num_examples, rng)
                  for layer in layers]

        zipped = safe_zip(layers, states)

        rval = OrderedDict(zipped)

        return rval

    def mcmc_steps(self, layer_to_state, theano_rng, layer_to_clamp=None,
                   num_steps=1):
        """
        .. todo::

            WRITEME
        """
        warnings.warn("DBM.mcmc_steps is deprecated. You should instead " +
                      "call DBM.sampling_procedure.sample, which defaults " +
                      "to what DBM.mcmc_steps used to do. This method will " +
                      "be removed on or after July 31, 2014.")
        return self.sampling_procedure.sample(layer_to_state, theano_rng,
                                              layer_to_clamp, num_steps)

    def get_sampling_updates(self, layer_to_state, theano_rng,
                             layer_to_clamp=None, num_steps=1,
                             return_layer_to_updated=False):
        """
        This method is for getting an updates dictionary for a theano function.

        It thus implies that the samples are represented as shared variables.
        If you want an expression for a sampling step applied to arbitrary
        theano variables, use the 'mcmc_steps' method. This is a wrapper around
        that method.

        Parameters
        ----------
        layer_to_state : dict
            Dictionary mapping the SuperDBM_Layer instances contained in
            self to shared variables representing batches of samples of them.
            (you can allocate one by calling self.make_layer_to_state)
        theano_rng : MRG_RandomStreams
            WRITEME
        layer_to_clamp : dict, optional
            Dictionary mapping layers to bools. If a layer is not in the
            dictionary, defaults to False. True indicates that this layer
            should be clamped, so we are sampling from a conditional
            distribution rather than the joint distribution
        num_steps : int, optional
            WRITEME
        return_layer_to_updated : bool, optional
            WRITEME

        Returns
        -------
        rval : dict
            Dictionary mapping each shared variable to an expression to
            update it. Repeatedly applying these updates does MCMC sampling.

        Notes
        -----
        The specific sampling schedule used by default is to sample all of the
        even-idexed layers of model.hidden_layers, then the visible layer and
        all the odd-indexed layers.
        """

        updated = self.sampling_procedure.sample(layer_to_state, theano_rng,
                                                 layer_to_clamp, num_steps)

        rval = OrderedDict()

        def add_updates(old, new):
            if isinstance(old, (list, tuple)):
                for old_elem, new_elem in safe_izip(old, new):
                    add_updates(old_elem, new_elem)
            else:
                rval[old] = new

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully
        # populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert key is self.visible_layer or key in self.hidden_layers

        for layer in [self.visible_layer] + self.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        # Translate update expressions into theano updates
        for layer in layer_to_state:
            old = layer_to_state[layer]
            new = updated[layer]
            if layer_to_clamp[layer]:
                assert new is old
            else:
                add_updates(old, new)

        assert isinstance(self.hidden_layers, list)

        if return_layer_to_updated:
            return rval, updated

        return rval

    def get_monitoring_channels(self, data):
        """
        .. todo::

            WRITEME
        """
        space, source = self.get_monitoring_data_specs()
        space.validate(data)
        X = data
        history = self.mf(X, return_history=True)
        q = history[-1]

        rval = OrderedDict()

        ch = self.visible_layer.get_monitoring_channels()
        for key in ch:
            rval['vis_' + key] = ch[key]

        for state, layer in safe_zip(q, self.hidden_layers):
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name + '_' + key] = ch[key]
            ch = layer.get_monitoring_channels_from_state(state)
            for key in ch:
                rval['mf_' + layer.layer_name + '_' + key] = ch[key]

        if len(history) > 1:
            prev_q = history[-2]

            flat_q = flatten(q)
            flat_prev_q = flatten(prev_q)

            mx = None
            for new, old in safe_zip(flat_q, flat_prev_q):
                cur_mx = abs(new - old).max()
                if new is old:
                    logger.error('{0} is {1}'.format(new, old))
                    assert False
                if mx is None:
                    mx = cur_mx
                else:
                    mx = T.maximum(mx, cur_mx)

            rval['max_var_param_diff'] = mx

            for layer, new, old in safe_zip(self.hidden_layers,
                                            q, prev_q):
                sum_diff = 0.
                for sub_new, sub_old in safe_zip(flatten(new), flatten(old)):
                    sum_diff += abs(sub_new - sub_old).sum()
                denom = self.batch_size * \
                    layer.get_total_state_space().get_total_dimension()
                denom = np.cast[config.floatX](denom)
                rval['mean_'+layer.layer_name+'_var_param_diff'] = \
                    sum_diff / denom

        return rval

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channel.

        This implementation returns specification corresponding to unlabeled
        inputs.
        """
        return (self.get_input_space(), self.get_input_source())

    def get_test_batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return self.batch_size

    def reconstruct(self, V):
        """
        .. todo::

            WRITEME
        """

        H = self.mf(V)[0]

        downward_state = self.hidden_layers[0].downward_state(H)

        recons = self.visible_layer.inpaint_update(
            layer_above=self.hidden_layers[0],
            state_above=downward_state,
            drop_mask=None, V=None)

        return recons

    def do_inpainting(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.setup_inference_procedure()
        return self.inference_procedure.do_inpainting(*args, **kwargs)
