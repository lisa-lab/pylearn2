"""
.. todo::

    WRITEME
"""
from pylearn2.compat import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.costs.autoencoder import GSNFriendlyCost
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip

class GSNCost(Cost):
    """
    Customizable cost class for GSNs.

    This class currently can only handle datasets with only one or two sets
    of vectors. The get_input_source and get_target_source methods on the model
    instance are called to get the names for the fields in the dataset.
    get_input_source() is used for the name of the first set of vectors and
    get_target_source() is used for the second set of vectors.

    The explicit use of get_input_source and get_target_source (and the
    non-existance of similar hooks) is what limits this class to learning
    the joint distribution between only 2 sets of vectors. The allow for more
    than 2 sets of vectors, the Model class would need to be modified, preferably
    in a way that allows reference to arbitrarily many sets of vectors within
    one dataset.

    Parameters
    ----------
    costs : list of (int, double, GSNFriendlyCost or callable) tuples
        The int component of each tuple is the index of the layer at
        which we want to compute this cost.
        The double component of the tuple is the coefficient to associate
        to with the cost.
        The GSNFriendlyCost instance is the cost that will be computed.
        If that is a callable rather than an instance of GSN friendly
        cost, it will be called with 2 arguments: the initial value
        followed by the reconstructed value.
        Costs must be of length 1 or 2 (explained in docstring for
        GSNCost class) and the meaning of the ordering of the costs
        parameter is explained in the docstring for the mode parameter.
    walkback : int
        How many steps of walkback to perform
    mode : str
        Must be either 'joint', 'supervised', or 'anti_supervised'.
        The terms "input layer" and "label layer" are used below in the
        description of the modes. The "input layer" refers to the layer
        at the index specified in the first tuple in the costs parameter,
        and the "label layer" refers to the layer at the index specified
        in the second tuple in the costs parameter.
        'joint' means setting all of the layers and calculating
        reconstruction costs.
        'supervised' means setting just the input layer and attempting to
        predict the label layer
        'anti_supervised' is attempting to predict the input layer given
        the label layer.
    """

    def __init__(self, costs, walkback=0, mode="joint"):
        super(GSNCost, self).__init__()
        self.walkback = walkback

        assert mode in ["joint", "supervised", "anti_supervised"]
        if mode in ["supervised", "anti_supervised"]:
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
        """
        Computes the total cost contribution from one layer given the full
        output of the GSN.

        Parameters
        ----------
        idx : int
            init_data and model_output both contain a subset of the layer \
            activations at each time step. This is the index of the layer we \
            want to evaluate the cost on WITHIN this subset. This is \
            generally equal to the idx of the cost function within the \
            GSNCost.costs list.
        costf : callable
            Function of two variables that computes the cost. The first \
            argument is the target value, and the second argument is the \
            predicted value.
        init_data : list of tensor_likes
            Although only the element at index "idx" is accessed/needed, this \
            parameter is a list so that is can directly handle the data \
            format from GSN.expr.
        model_output : list of list of tensor_likes
            The output of GSN.get_samples as called by GSNCost.expr.
        """
        total = 0.0
        for step in model_output:
            total += costf(init_data[idx], step[idx])

        # normalize for number of steps
        return total / len(model_output)

    def _get_samples_from_model(self, model, data):
        """
        .. todo::

            WRITEME properly
        
        Handles the different GSNCost modes.
        """
        layer_idxs = [idx for idx, _, _ in self.costs]
        zipped = safe_zip(layer_idxs, data)
        if self.mode == "joint":
            use = zipped
        elif self.mode == "supervised":
            # don't include label layer
            use = zipped[:1]
        elif self.mode == "anti_supervised":
            # don't include features
            use = zipped[1:]
        else:
            raise ValueError("Unknown mode \"%s\" for GSNCost" % self.mode)

        return model.get_samples(use,
                                 walkback=self.walkback,
                                 indices=layer_idxs)

    def expr(self, model, data):
        """
        Parameters
        ----------
        model : GSN object
            WRITEME
        data : list of tensor_likes
            Data must be a list or tuple of the same length as self.costs. \
            All elements in data must be a tensor_like (cannot be None).

        Returns
        -------
        y : tensor_like
            The actual cost that is backpropagated on.
        """
        self.get_data_specs(model)[0].validate(data)
        output = self._get_samples_from_model(model, data)

        total = 0.0
        for cost_idx, (_, coeff, costf) in enumerate(self.costs):
            total += (coeff *
                      self._get_total_for_cost(cost_idx, costf, data, output))

        coeff_sum = sum(coeff for _, coeff, _ in self.costs)

        # normalize for coefficients on each cost
        return total / coeff_sum

    def get_monitoring_channels(self, model, data, **kwargs):
        """
        .. todo::

            WRITEME properly
        
        Provides monitoring of the individual costs that are being added together.

        This is a very useful method to subclass if you need to monitor more
        things about the model.
        """
        self.get_data_specs(model)[0].validate(data)

        rval = OrderedDict()

        # if there's only 1 cost, then no need to split up the costs
        if len(self.costs) > 1:
            output = self._get_samples_from_model(model, data)

            rval['reconstruction_cost'] =\
                self._get_total_for_cost(0, self.costs[0][2], data, output)

            rval['classification_cost'] =\
                self._get_total_for_cost(1, self.costs[1][2], data, output)

        return rval

    def get_data_specs(self, model):
        """
        .. todo::

            WRITEME
        """
        # get space for layer i of model
        get_space = lambda i: (model.aes[i].get_input_space() if i==0
                               else model.aes[i - 1].get_output_space())

        # get the spaces for layers that we have costs at
        spaces = map(lambda c: get_space(c[0]), self.costs)

        sources = [model.get_input_source()]
        if len(self.costs) == 2:
            sources.append(model.get_target_source())

        return (CompositeSpace(spaces), tuple(sources))
