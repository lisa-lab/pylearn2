from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import GSNFriendlyCost
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip

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
        self.walkback = walkback

        assert len(costs) in [1, 2], "This is (hopefully) a temporary restriction"
        assert len(set(c[0] for c in costs)) == len(costs), "Must have only" +\
            " one cost function per index"
        self.costs = costs

        # convert GSNFriendCost instances into just callables
        for i, cost_tup in enumerate(self.costs):
            if isinstance(cost_tup[2], GSNFriendlyCost):
                mutable = list(cost_tup)
                mutable[2] = cost_tup[2].cost
                self.costs[i] = tuple(mutable)
            else:
                assert callable(cost_tup[2])

    def expr(self, model, data):
        """
        Parameters
        ----------
        model : GSN object
        data : list of tensor_likes
            data must be a list or tuple of the same length as self.costs. All
            elements in data must be a tensor_like (cannot be None).
        """

        layer_idxs = [idx for idx, _, _ in self.costs]
        output = model.get_samples(safe_zip(layer_idxs, data),
                                   walkback=self.walkback, indices=layer_idxs)

        total = 0.0
        for cost_idx, (_, coeff, costf) in enumerate(self.costs):
            cost_total = 0.0
            for step in output:
                cost_total += costf(data[cost_idx], step[cost_idx])
            total += coeff * cost_total
        return total

    def get_data_specs(self, model):
        # get space for layer i
        get_space = lambda i: (model.aes[i].get_input_space() if i==0
                               else model.aes[i - 1].get_output_space())

        spaces = map(lambda c: get_space(c[0]), self.costs)
        sources = [model.get_input_source()]
        if len(self.costs) == 2:
            sources.append(model.get_output_source())

        return (CompositeSpace(spaces), tuple(sources))

