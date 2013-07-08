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
        assert len(data) == len(costs)

        indices = [c[0] for c in self.costs]

        non_none = filter(lambda i: data[i] is not None, indices)

        # make idx, value pairs
        data = zip(indices, data)

        output = model.get_samples(data, walkback=self.walkback,
                                   indices=non_none)

        layer_cost = lambda activations: sum(
            self.costs[i][1] * self.costs[i][2](data[i], activations[i])
            for i in xrange(len(self.costs))
        )

        return sum(map(layer_cost, output))

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


