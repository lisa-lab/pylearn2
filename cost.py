"""
Cost classes: classes that encapsulate the cost evaluation for the DAE
training criterion.
"""
# Standard library imports
from itertools import izip

# Third-party imports
from theano import tensor

class SupervisedCost(object):
    """
    A cost object is allocated in the same fashion as other
    objects in this file, with a 'conf' dictionary (or object
    supporting __getitem__) containing relevant hyperparameters.
    """
    def __init__(self, conf, model):
        self.conf = conf
        # TODO: Do stuff depending on conf parameters (for example
        # use different cross-entropy if act_end == "tanh" or not)

    def __call__(self, *inputs):
        """Symbolic expression denoting the reconstruction error."""
        raise NotImplementedError()

class MeanSquaredError(SupervisedCost):
    """
    Symbolic expression for mean-squared error between the input and the
    denoised reconstruction.
    """
    def __call__(self, prediction, target):
        msq = lambda p, t: ((p - t)**2).sum(axis=1).mean()
        if isinstance(prediction, tensor.Variable):
            return msq(prediction, target)
        else:
            pairs = izip(prediction, target)
            # TODO: Think of something more sensible to do than sum(). On one
            # hand, if we're treating everything in parallel it should return
            # a list. On the other, we need a scalar for everything else to
            # work.

            # This will likely get refactored out into a "costs" module or
            # something like that.
            return sum([msq(p, t) for p, t in pairs])

class CrossEntropy(SupervisedCost):
    """
    Symbolic expression for elementwise cross-entropy between input
    and reconstruction. Use for binary-valued features (but not for,
    e.g., one-hot codes).
    """
    def __call__(self, prediction, target):
        ce = lambda x, z: x * tensor.log(z) + (1 - x) * tensor.log(1 - z)
        if isinstance(prediction, tensor.Variable):
            return ce(prediction, target)
        pairs = izip(prediction, target)
        return sum([ce(p, t).sum(axis=1).mean() for p, t in pairs])

##################################################
def get(str):
    """ Evaluate str into a cost object, if it exists """
    obj = globals()[str]
    if issubclass(obj, Cost):
        return obj
    else:
        raise NameError(str)
