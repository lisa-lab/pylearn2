from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import GSNFriendlyCost
from pylearn2.space import CompositeSpace

class GSNCost(Cost):
    def __init__(self, costs, walkback=0):
        """
        Parameters
        ----------
        costs : list of (int, double, GSNFriendlyCost or callable) tuples
            The int component of each tuple is the index of the layer at which
            we want to compute this cost.
            The double component of the tuple is the coefficient to associate
            to with the cost.
            The GSNFriendlyCost instance is the cost that will be computed. If
            that is a callable rather than an instance of GSN friendly cost,
            it will be called with 2 arguments: the initial value and the
            reconstructed value.
        walkback : int
            how many steps of walkback to perform
        """
        super(GSNCost, self).__init__()
        self.costs = costs

        # assuming full supervised training
        self.set_active_layers([c[0] for c in self.costs])

        # convert GSNFriendCost instances into just callables
        for i, cost_tup in enumerate(self.costs):
            if isinstance(cost_tup[2], GSNFriendlyCost):
                mutable = list(cost_tup)
                mutable[2] = cost_tup[2].cost
                self.costs[i] = tuple(mutable)

        self.walkback = walkback

    def set_active_layers(layer_idxs):
        """
        Informs the GSNCost which layers will be set at first.

        This function can be called to switch between supervised and unsupervised
        training.

        Parameters
        ----------
        layers : list of int
            Indicates which layers will be set for forthcoming training. This
            list MUST be sorted in the same order as the indexes for the costs
            passed into to __init__.

        Note
        ----
        GSNCost.expr currently works without the contraint that only some layers
        can be set at once. This constraint is necessary to implement
        GSNCost.get_data_specs. Either GSNCost.expr should be restricted (would
        make the code simpler) or the Space model should allow union types.
        """
        assert len(layer_idxs) > 0
        self._cost_subset = filter(lambda c: c[0] in layer_idxs, self.costs)

    def expr(self, model, data):
        """
        Parameters
        ----------
        model : GSN object
        data : list of tensor_likes
            data must be a list of the same length as self._cost_subset, which
            defaults to all of self.costs and is specified by the indexes passed
            to self._set_active_layers. All elements in data must be tensor_likes
            (ie they cannot be None).
        """
        assert len(data) == len(self._cost_subset)

        zerof = lambda x, y: 0.0

        # just treat every layer as if it had a cost function (where most of them
        # are always 0)
        expanded_data = [None] * (len(model.aes) + 1)
        expanded_costs = [(0.0, zerof)] * (len(model.aes) + 1)

        for cost_idx, layer_idx, coeff, costf in enumerate(self._cost_subset):
            expanded_data[layer_idx] = data[cost_idx]

            assert expanded_data[layer_idx] is not None
            expanded_costs[layer_idx] = (coeff, costf)

        cost_indices = [c[0] for c in self._cost_subset]

        # get all of the output except at first time step
        output = model.get_samples(zip(cost_indices, data), walkback=0,
                                   indices=range(len(expanded_data)))

        ec = expanded_costs

        # cost occuring at step idx
        step_sum = lambda idx: sum(
            ec[i][0] * ec[i][1](expanded_data[i], output[idx][i])
            for i in xrange(len(ec))
        )

        return sum(map(step_sum, xrange(len(output))))

    def get_data_specs(self, model):
        if len(self._cost_subset) > 1:
            spaces = []
            for idx, _, _ in self._cost_subset:
                if idx == 0:
                    spaces.append(model.aes[0].get_input_space())
                else:
                    spaces.append(model.aes[idx - 1].get_output_space())
            spaces = CompositeSpace(spaces)
            sources = (model.get_input_source(), model.get_target_source())
            return (spaces, sources)
        else:
            idx = self._layer_idx[0]
            if idx == 0:
                space = model.aes[0].get_input_space()
            else:
                space = model.aes[idx - 1].get_output_space()
            return (space, model.get_target_source())

