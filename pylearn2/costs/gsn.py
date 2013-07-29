from theano.compat.python2x import OrderedDict

from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import GSNFriendlyCost
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip

class GSNCost(Cost):
    def __init__(self, costs, walkback=0, mode="joint"):
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
        mode : str
            Should be either 'joint' or 'supervised'. joint means setting
            all of the layers and calculating reconstruction costs. 'supervised'
            means setting just the input layer (the first cost) and attempting
            to predict the label layer (the second cost).

        Note
        ----
        As of now, the costs list can contain only 1 or 2 elements. If it contains
        2 elements, the first element must be for the input to the model and the
        second must be the output/labels.
        """
        super(GSNCost, self).__init__()
        self.walkback = walkback

        assert mode in ["joint", "supervised"]
        if mode == "supervised":
            assert len(costs) == 2
        self.mode = mode

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

    @staticmethod
    def _get_total_for_cost(idx, costf, init_data, model_output):
        total = 0.0
        for step in model_output:
            total += costf(init_data[idx], step[idx])
        return total / len(model_output)

    def _get_samples_from_model(self, model, data):
        layer_idxs = [idx for idx, _, _ in self.costs]
        zipped = safe_zip(layer_idxs, data)
        if self.mode == "joint":
            return model.get_samples(zipped,
                                     walkback=self.walkback,
                                     indices=layer_idxs)
        else:
            # don't include the label layer
            return model.get_samples(zipped[:1],
                                     walkback=self.walkback,
                                     indices=layer_idxs)

    def expr(self, model, data):
        """
        Parameters
        ----------
        model : GSN object
        data : list of tensor_likes
            data must be a list or tuple of the same length as self.costs. All
            elements in data must be a tensor_like (cannot be None).
        """
        self.get_data_specs(model)[0].validate(data)
        output = self._get_samples_from_model(model, data)

        total = 0.0
        for cost_idx, (_, coeff, costf) in enumerate(self.costs):
            total += (coeff *
                      self._get_total_for_cost(cost_idx, costf, data, output))

        coeff_sum = sum(coeff for _, coeff, _ in self.costs)

        # little bit of normalization
        return total / coeff_sum

    def get_monitoring_channels(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        output = self._get_samples_from_model(model, data)

        rval = OrderedDict()
        rval['reconstruction_cost'] =\
            self._get_total_for_cost(0, self.costs[0][2], data, output)

        if len(self.costs) == 2:
            rval['classification_cost'] =\
                self._get_total_for_cost(1, self.costs[1][2], data, output)

        return rval

    def get_data_specs(self, model):
        # get space for layer i
        get_space = lambda i: (model.aes[i].get_input_space() if i==0
                               else model.aes[i - 1].get_output_space())

        spaces = map(lambda c: get_space(c[0]), self.costs)
        sources = [model.get_input_source()]
        if len(self.costs) == 2:
            sources.append(model.get_target_source())

        return (CompositeSpace(spaces), tuple(sources))


