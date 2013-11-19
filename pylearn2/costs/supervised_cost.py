import warnings
warnings.warn("The pylearn2.supervised_cost module is deprecated."
        "Its name was confusing because it did not actually define"
        "SupervisedCost, which is and was defined in cost.py")

# preserve old import in case anyone was referring to SupervisedCost
# by this location
from pylearn2.costs.cost import Cost
# import the only class that was defined here, so old code can still
# import it
from pylearn2.costs.cost import CrossEntropy
from pylearn2.space import CompositeSpace
import theano.tensor as T


class CrossEntropy(Cost):
    supervised = True

    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)

class NegativeLogLikelihood(Cost):
    """
    Represents the mean negative log-likelihood of a model's output, provided
    the target Y is one-hot encoded. Equivalent to

        .. math::

        cost = - \frac{1}{N} \sum_{i=1}^N log(p(Y = y^{(i)} | x^{(i)}, \theta))

    We compute the mean of the negative log-likelihood instead of the NLL
    itself for the sake of simplicity and to make the cost more invariant to
    the dataset's size.
    """
    supervised = True

    def expr(self, model, data):
        """
        Returns the mean negative log-likelihood of a model for input X given
        a one-hot encoded target Y.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the negative
            log-likelihood
        data : tuple of tensor_like
            (input to the model, one-hot encoded target)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        return (-Y * T.log(model.fprop(X))).sum(axis=1).mean()

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
