"""
This module contains cost functions to use with Boltzmann machines
(pylearn2.models.boltzmann_machine).
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict
import numpy

from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_zip


class BaseCD(Cost):
    def __init__(self, num_chains, num_gibbs_steps):
        self.num_chains = num_chains
        self.num_gibbs_steps = num_gibbs_steps
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        self.numpy_rng = numpy.random.RandomState([2012, 10, 17])

    def expr(self, model, data):
        """
        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)

        return None

    def get_monitoring_channels(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()

        return rval

    def get_gradients(self, model, data):
        self.get_data_specs(model)[0].validate(data)

        X = [data[:, partition] for partition in model.data_partition]
        Y = None

        pos_phase_grads, pos_updates = self._get_positive_phase(model, X, Y)

        neg_phase_grads, neg_updates = self._get_negative_phase(model, X, Y)

        updates = OrderedDict()
        for key, val in pos_updates.items():
            updates[key] = val
        for key, val in neg_updates.items():
            updates[key] = val

        gradients = OrderedDict()
        for param in list(pos_phase_grads.keys()):
            gradients[param] = neg_phase_grads[param] + pos_phase_grads[param]

        return gradients, updates

    def _get_standard_neg(self, model, X, Y=None):
        layer_to_state = model.make_layer_to_state(
            batch_size=self.num_chains,
            numpy_rng=self.numpy_rng
        )

        layer_to_updated_state = model.sample(
            layer_to_state=layer_to_state,
            theano_rng=self.theano_rng,
            n_steps=self.num_gibbs_steps
        )

        updates = OrderedDict()
        for old_state, new_state in safe_zip(layer_to_state.values(),
                                             layer_to_updated_state.values()):
            updates[old_state] = new_state

        params = list(model.get_params())

        energy = model.energy(layer_to_updated_state).mean()

        samples = layer_to_updated_state.values()
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = OrderedDict(
            safe_zip(params, theano.tensor.grad(-energy, params,
                                                consider_constant=samples,
                                                disconnected_inputs='ignore'))
        )

        return neg_phase_grads, updates

    def _get_variational_pos(self, model, X, Y):
        layer_to_symbolic_state = model.make_layer_to_symbolic_state(
            batch_size=self.num_chains,
            theano_rng=self.theano_rng
        )

        for visible_layer, state in safe_zip(model.visible_layers, X):
            layer_to_symbolic_state[visible_layer] = state

        layer_to_updated_state = model.variational_inference(
            layer_to_state=layer_to_symbolic_state,
            n_steps=self.num_gibbs_steps
        )

        params = list(model.get_params())

        energy = model.energy(layer_to_updated_state).mean()

        samples = layer_to_updated_state.values()
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        gradients = OrderedDict(
            safe_zip(params, theano.tensor.grad(energy, params,
                                                consider_constant=samples,
                                                disconnected_inputs='ignore'))
        )

        return gradients, OrderedDict()


class VariationalPCD(BaseCD):
    """
    An intractable cost representing the variational upper bound on the
    negative log likelihood of a Boltzmann machine. The gradient of this bound
    is computed using a persistent markov chain.

    .. todo::

        Add citation to Tieleman paper, Younes paper
    """
    def _get_positive_phase(self, model, X, Y=None):
        return self._get_variational_pos(model, X, Y)

    def _get_negative_phase(self, model, X, Y=None):
        return self._get_standard_neg(model, X, Y)

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())
