from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import GSNFriendlyCost

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

        # convert GSNFriendCost instances into just callables
        for i, cost_tup in enumerate(self.costs):
            if isinstance(cost_tup[2], GSNFriendlyCost):
                mutable = list(cost_tup)
                mutable[2] = cost_tup[2].cost
                self.costs[i] = tuple(mutable)

        self.walkback = walkback

    def expr(self, model, data):
        """
        Parameters
        ----------
        model : GSN object
        data : list of tensor_likes
            data must be a list of the same length as self.costs. Each element
            in data can either be a tensor_like or None. The value at data[i]
            will be placed in the initial activation array at position
            self.costs[i][0] (the layer_idx component). This means that the
            ordering on the data is identical to the order of the cost functions
            that this GSNCost instance was created with.
        """
        assert len(data) == len(self.costs)
        zerof = lambda x, y: 0.0

        # just treat every layer as if it had a cost function (where most of them
        # are always 0)
        expanded_data = [None] * (len(model.aes) + 1)
        expanded_costs = [(0.0, zerof)] * (len(model.aes) + 1)

        for cost_idx, layer_idx, coeff, costf in enumerate(self.costs):
            expanded_data[layer_idx] = data[cost_idx]

            # we only want to calculate costs where data is not None
            if expanded_data[layer_idx] is not None:
                expanded_costs[layer_idx] = (coeff, costf)

        for i in xrange(len(expanded_data)):
            if expanded_data[i] is None:
                assert expanded_costs[i][1] is zerof

        cost_indices = [c[0] for c in self.costs]

        # get all of the output except at first time step
        output = model.get_samples(zip(cost_indices, data), walkback=0,
                                   indices=range(len(expanded_data)))

        ec = expanded_costs

        # cost occuring at step idx
        step_sum = lambda idx: sum(
            ec[i][0] * ec[i][1](expanded_data[i], output[idx][i])
            for i in xrange(len(ec))
        )

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


