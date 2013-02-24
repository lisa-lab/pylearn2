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
import theano.tensor as T

class MeanSquareError(Cost):
    supervised = True
    
    def __call__(self, model, X, Y, **kwargs):
        return T.sqr(Y - model.fprop(X, **kwargs)).sum(axis=1).mean()
        
class SumSquareError(Cost):
    supervised = True
    
    def __call__(self, model, X, Y, **kwargs):
        return T.sqr(Y - model.fprop(X, **kwargs)).sum(axis=1)

class CrossEntropy(Cost):
    supervised = True

    def __call__(self, model, X, Y, **kwargs):
        model_output = model.fprop(X, **kwargs)
        return (-Y * T.log(model_output) - \
                (1 - Y) * T.log(1 - model_output)).sum(axis=1).mean()


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

    def __call__(self, model, X, Y):
        """
        Returns the mean negative log-likelihood of a model for input X given
        a one-hot encoded target Y.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the negative
            log-likelihood
        X : tensor_like
            input to the model
        Y : tensor_like
            one-hot encoded target
        """
        return (-Y * T.log(model(X))).sum(axis=1).mean()
