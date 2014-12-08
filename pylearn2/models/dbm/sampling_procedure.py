"""
Classes that implement different sampling algorithms for DBMs.
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

from theano.compat.six.moves import xrange
from pylearn2.compat import OrderedDict
from pylearn2.utils import py_integer_types


class SamplingProcedure(object):
    """
    Procedure for sampling from a DBM.
    """

    def set_dbm(self, dbm):
        """
        Associates the SamplingProcedure with a specific DBM.

        Parameters
        ----------
        dbm : pylearn2.models.dbm.DBM instance
            The model to perform sampling from.
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
            Random number generator
        layer_to_clamp : dict, optional
            Maps Layers to bools. If a layer is not in the dictionary,
            defaults to False. True indicates that this layer should be
            clamped, so we are sampling from a conditional distribution
            rather than the joint distribution.
        num_steps : int, optional
            Steps of the sampling procedure. It samples for `num_steps`
            times and return the last sample.

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
    The specific sampling schedule used to sample all of the even-idexed
    layers of model.hidden_layers, then the visible layer and all the
    odd-indexed layers.
    """

    def sample(self, layer_to_state, theano_rng, layer_to_clamp=None,
               num_steps=1):
        """
        .. todo::

            WRITEME
        """
        # Validate num_steps
        assert isinstance(num_steps, py_integer_types)
        assert num_steps > 0

        # Implement the num_steps > 1 case by repeatedly calling the
        # num_steps == 1 case
        if num_steps != 1:
            for i in xrange(num_steps):
                layer_to_state = self.sample(layer_to_state, theano_rng,
                                             layer_to_clamp, num_steps=1)
            return layer_to_state

        # The rest of the function is the num_steps = 1 case
        # Current code assumes this, though we could certainly relax this
        # constraint
        assert len(self.dbm.hidden_layers) > 0

        # Validate layer_to_clamp / make sure layer_to_clamp is a fully
        # populated dictionary
        if layer_to_clamp is None:
            layer_to_clamp = OrderedDict()

        for key in layer_to_clamp:
            assert (key is self.dbm.visible_layer or
                    key in self.dbm.hidden_layers)

        for layer in [self.dbm.visible_layer] + self.dbm.hidden_layers:
            if layer not in layer_to_clamp:
                layer_to_clamp[layer] = False

        # Assemble the return value
        layer_to_updated = OrderedDict()

        for i, this_layer in list(enumerate(self.dbm.hidden_layers))[::2]:
            # Iteration i does the Gibbs step for hidden_layers[i]

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            if i == 0:
                layer_below = self.dbm.visible_layer
            else:
                layer_below = self.dbm.hidden_layers[i-1]
            state_below = layer_to_state[layer_below]
            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.dbm.hidden_layers):
                layer_above = self.dbm.hidden_layers[i + 1]
                state_above = layer_to_state[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                state_above = None
                layer_above = None

            if layer_to_clamp[this_layer]:
                this_state = layer_to_state[this_layer]
                this_sample = this_state
            else:
                # Compute the Gibbs sampling update
                # Sample the state of this layer conditioned
                # on its Markov blanket (the layer above and
                # layer below)
                this_sample = this_layer.sample(state_below=state_below,
                                                state_above=state_above,
                                                layer_above=layer_above,
                                                theano_rng=theano_rng)

            layer_to_updated[this_layer] = this_sample

        #Sample the visible layer
        vis_state = layer_to_state[self.dbm.visible_layer]
        if layer_to_clamp[self.dbm.visible_layer]:
            vis_sample = vis_state
        else:
            first_hid = self.dbm.hidden_layers[0]
            state_above = layer_to_updated[first_hid]
            state_above = first_hid.downward_state(state_above)

            vis_sample = self.dbm.visible_layer.sample(state_above=state_above,
                                                       layer_above=first_hid,
                                                       theano_rng=theano_rng)
        layer_to_updated[self.dbm.visible_layer] = vis_sample

        # Sample the odd-numbered layers
        for i, this_layer in list(enumerate(self.dbm.hidden_layers))[1::2]:

            # Get the sampled state of the layer below so we can condition
            # on it in our Gibbs update
            layer_below = self.dbm.hidden_layers[i-1]

            # We want to sample from each conditional distribution
            # ***sequentially*** so we must use the updated version
            # of the state for the layers whose updates we have
            # calculcated already, in layer_to_updated.
            # If we used the original value from
            # layer_to_state
            # then we would sample from each conditional
            # ***simultaneously*** which does not implement MCMC
            # sampling.
            state_below = layer_to_updated[layer_below]

            state_below = layer_below.upward_state(state_below)

            # Get the sampled state of the layer above so we can condition
            # on it in our Gibbs step
            if i + 1 < len(self.dbm.hidden_layers):
                layer_above = self.dbm.hidden_layers[i + 1]
                state_above = layer_to_updated[layer_above]
                state_above = layer_above.downward_state(state_above)
            else:
                state_above = None
                layer_above = None

            if layer_to_clamp[this_layer]:
                this_state = layer_to_state[this_layer]
                this_sample = this_state
            else:
                # Compute the Gibbs sampling update
                # Sample the state of this layer conditioned
                # on its Markov blanket (the layer above and
                # layer below)
                this_sample = this_layer.sample(state_below=state_below,
                                                state_above=state_above,
                                                layer_above=layer_above,
                                                theano_rng=theano_rng)

            layer_to_updated[this_layer] = this_sample

        # Check that all layers were updated
        assert all([layer in layer_to_updated for layer in layer_to_state])
        # Check that we didn't accidentally treat any other object as a layer
        assert all([layer in layer_to_state for layer in layer_to_updated])
        # Check that clamping worked
        assert all([(layer_to_state[layer] is layer_to_updated[layer]) ==
                    layer_to_clamp[layer] for layer in layer_to_state])

        return layer_to_updated
