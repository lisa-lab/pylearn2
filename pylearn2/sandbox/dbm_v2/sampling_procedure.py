"""
.. todo::

    WRITEME
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

from theano.compat import OrderedDict
from pylearn2.utils import py_integer_types


class SamplingProcedure(object):
    """
    Procedure for sampling from a DBM.
    """

    def set_dbm(self, dbm):
        """
        .. todo::

            WRITEME
        """
        self.dbm = dbm

    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None,
               num_steps=1):
        """
        Samples from self.dbm using `layer_to_state` as starting values.

        Parameters
        ----------
        layer_to_state : dict
            Maps the DBM's Layer instances to theano variables representing
            batches of samples of them.
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            WRITEME
        layer_to_clamp : dict, optional
            Maps Layers to bools. If a layer is not in the dictionary,
            defaults to False. True indicates that this layer should be
            clamped, so we are sampling from a conditional distribution
            rather than the joint distribution.
        num_steps : int, optional
            WRITEME

        Returns
        -------
        layer_to_updated_state : dict
            Maps the DBM's Layer instances to theano variables representing
            batches of updated samples of them.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "sample.")


class GibbsEvenOdd(SamplingProcedure):
    """
    Even-odd Gibbs sampling.
    The specific sampling schedule used to sample all of the even-idexed
    layers of model.hidden_layers, then the visible layer and all the
    odd-indexed layers.
    """

    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None, num_steps=1):
        """
        Samples from self.dbm using `layer_to_state` as starting values.

        Parameters
        ----------
        layer_to_state : dict
            Maps the DBM's Layer instances to theano variables representing
            batches of samples of them.
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            WRITEME
        layer_to_clamp : dict, optional
            Maps Layers to bools. If a layer is not in the dictionary,
            defaults to False. True indicates that this layer should be
            clamped, so we are sampling from a conditional distribution
            rather than the joint distribution.
        num_steps : int, optional
            Steps to sample the odd units. Evens are always done num_steps+1
            TODO: this is a hack to make CD work for RBM for now.

        Returns
        -------
        layer_to_updated_state : dict
            Maps the DBM's Layer instances to theano variables representing
            batches of updated samples of them.
        """

        assert isinstance(num_steps, py_integer_types) and num_steps >= 0

        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert key in self.dbm.hidden_layers + [self.dbm.visible_layer]

        # Set layer to clamps.
        for layer in self.dbm.hidden_layers + [self.dbm.visible_layer]:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        # Set all updates to the initial states.
        # For now, assert that the layer_to_state is full, but we will change
        # this later. Right now the cost function initializes everything.
        for layer in [self.dbm.visible_layer] + self.dbm.hidden_layers:
            assert layer in layer_to_state
        layer_to_updated = OrderedDict()
        for layer in [self.dbm.visible_layer] + self.dbm.hidden_layers:
            layer_to_updated[layer] = layer_to_state[layer]

        def update(i, this_layer):
            if layer_to_clamp[this_layer]:
                return

            # States and layers below
            if i == -1: # visible layer, will change.
                layer_below = None
                state_below = None
            else:
                if i == 0:
                    layer_below = self.dbm.visible_layer
                elif i > 0:
                    layer_below = self.dbm.hidden_layers[i-1]
                state_below = layer_to_updated[layer_below]
                state_below = layer_below.upward_state(state_below)

            # States and layers above
            if i + 1 < len(self.dbm.hidden_layers):
                layer_above = self.dbm.hidden_layers[i + 1]
                state_above = layer_to_updated[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                layer_above = None
                state_above = None

            this_sample = this_layer.sample(state_below=state_below,
                                            state_above=state_above,
                                            layer_above=layer_above,
                                            theano_rng=theano_rng)

            layer_to_updated[this_layer] = this_sample

        evens = list(enumerate(self.dbm.hidden_layers))[::2]
        # Odds are the visible layer plus the odd hidden layers
        odds = [(-1, self.dbm.visible_layer)] + list(enumerate(self.dbm.hidden_layers))[1::2]

        update_count = OrderedDict((layer, 0) for l in [self.dbm.visible_layer] + self.dbm.hidden_layers)

        for i, this_layer in evens:
            update(i, this_layer)
        for s in xrange(num_steps):
            for i, this_layer in odds:
                update(i, this_layer)
            for i, this_layer in evens:
                update(i, this_layer)

        # Check that all layers were updated
        assert all([layer in layer_to_updated for layer in layer_to_state])
        # Check that we didn't accidentally treat any other object as a layer
        assert all([layer in layer_to_state for layer in layer_to_updated])
        # Check that clamping worked
        assert all([(layer_to_state[layer] is layer_to_updated[layer]) ==
                    layer_to_clamp[layer] for layer in layer_to_state])

        return layer_to_updated