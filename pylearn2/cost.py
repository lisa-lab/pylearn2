"""
Cost classes: classes that encapsulate the cost evaluation for various
training criteria.
"""
# Standard library imports
from itertools import imap

# Third-party imports
from theano import tensor


class SupervisedCost(object):
    def __init__(self, model=None):
        # TODO: Do stuff depending on model hyperparameters (for example
        # use different cross-entropy if act_enc == "tanh" or not)
        self.model = model

    def __call__(self, *inputs):
        """Symbolic expression denoting the error.

        If the inputs contain data regarding a mini-batch of examples,
        (for instance, predictions and targets for a mini-batch),
        then the output will contain the error for each of the examples.
        """
        raise NotImplementedError()


class SquaredError(SupervisedCost):
    """
    Symbolic expression for squared error between the target
    and a prediction.
    """
    def __call__(self, prediction, target):
        msq = lambda p, t: tensor.sum((p - t) ** 2, axis=1)
        if isinstance(prediction, tensor.Variable):
            return msq(prediction, target)
        else:
            # TODO: Think of something more sensible to do than sum(). On one
            # hand, if we're treating everything in parallel it should return
            # a list. On the other, we need a scalar for everything else to
            # work.

            # This will likely get refactored out into a "costs" module or
            # something like that.
            return sum(imap(msq, prediction, target))


class BinaryCrossEntropy(SupervisedCost):
    """
    Symbolic expression for elementwise cross-entropy between target
    and prediction. Use for binary-valued features (but not for,
    e.g., one-hot codes).
    """
    def __call__(self, prediction, target):
        ce = lambda x, z: (-((x * tensor.log(z)
                      + (1 - x) * tensor.log(1 - z)).sum(axis=1)))

        if isinstance(prediction, tensor.Variable):
            return ce(prediction, target)
        return sum(
            imap(lambda p, t: ce(p, t).sum(axis=1).mean(), prediction, target)
        )


class OneHotCrossEntropy(SupervisedCost):
    """
    Symbolic expression for the NLL of a multinomial classification prediction

    Use when the prediction is a vector of probabilities, and the target
    contains the index of the correct label.

    .. math::

        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
        \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

    Returns the mean over a minibatch.
    """
    def __call__(self, prediction, target):
        """TODO: Document me."""
        # Number of rows in target, i.e., number of examples in the minibatch.
        batch_size = target.shape[0]
        # tensor.arange(batch_size) is a symbolic vector which will contain
        # [0,1,2,... batch_size-1]
        # log(prediction) is a matrix of Log-Probabilities (call it LP) with
        # one row per example and one column per class.
        lp = tensor.log(prediction)
        # LP[arange(batch_size),target] is a vector v containing
        # [LP[0,target[0]], ..., LP[batch_size-1,target[batch_size-1]]]
        v = lp[tensor.arange(batch_size), target]
        # We return the negative log-likelihood of each element of
        # the minibatch
        return -v
