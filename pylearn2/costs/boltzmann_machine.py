"""
This module contains cost functions to use with Boltzmann machines
(pylearn2.models.boltzmann_machine).
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"


class BaseCD(Cost):
    def __init__(self, num_chains, num_gibbs_steps):
        self.num_chains = num_chains
        self.num_gibbs_steps = num_gibbs_steps
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)

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
        if self.supervised:
            X, Y = data
            assert Y is not None
        else:
            X = data
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

    def _get_standard_neg(self, model, layer_to_chains):
        params = list(model.get_params())

        expected_energy_p = model.energy(
            layer_to_chains[model.visible_layer],
            [layer_to_chains[layer] for layer in model.hidden_layers]
        ).mean()

        samples = layer_to_chains.values()
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = OrderedDict(
            safe_zip(params, T.grad(-expected_energy_p, params,
                                    consider_constant=samples,
                                    disconnected_inputs='ignore'))
        )
        return neg_phase_grads

    def _get_variational_pos(self, model, X, Y):

        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(
            safe_zip(params, T.grad(expected_energy_q,
                                    params,
                                    consider_constant=variational_params,
                                    disconnected_inputs='ignore'))
        )
        return gradients


class VariationalPCD(BaseCD):
    """
    An intractable cost representing the variational upper bound on the
    negative log likelihood of a Boltzmann machine. The gradient of this bound
    is computed using a persistent markov chain.

    .. todo::

        Add citation to Tieleman paper, Younes paper
    """
    def _get_positive_phase(self, model, X, Y=None):
        return self._get_variational_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """
        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(
            layer_to_chains,
            self.theano_rng, num_steps=self.num_gibbs_steps,
            return_layer_to_updated=True)

        neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, updates

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())
