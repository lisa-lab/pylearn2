"""
Cost classes: classes that encapsulate the cost evaluation for the DAE
training criterion.
"""
# Standard library imports
from itertools import izip

# Third-party imports
from theano import tensor

class Cost(object):
    """
    A cost object is allocated in the same fashion as other
    objects in this file, with a 'conf' dictionary (or object
    supporting __getitem__) containing relevant hyperparameters.
    """
    def __init__(self, conf, da):
        self.reconstruction = da.reconstruction
        self.conf = conf
        # TODO: Do stuff depending on conf parameters (for example
        # use different cross-entropy if act_end == "tanh" or not)

    def __call__(self, inputs):
        """Symbolic expression denoting the reconstruction error."""
        raise NotImplementedError()

class MeanSquaredError(Cost):
    """
    Symbolic expression for mean-squared error between the input and the
    denoised reconstruction.
    """
    def __call__(self, inputs):
        pairs = izip(inputs, self.reconstruction(inputs))
        # TODO: Think of something more sensible to do than sum(). On one
        # hand, if we're treating everything in parallel it should return
        # a list. On the other, we need a scalar for everything else to
        # work.

        # This will likely get refactored out into a "costs" module or
        # something like that.
        return sum([((inp - rec) ** 2).sum(axis=1).mean() for inp, rec in pairs])

class CrossEntropy(Cost):
    """
    Symbolic expression for elementwise cross-entropy between input
    and reconstruction. Use for binary-valued features (but not for,
    e.g., one-hot codes).
    """
    def __call__(self, inputs):
        pairs = izip(inputs, self.reconstruction(inputs))
        ce = lambda x, z: x * tensor.log(z) + (1 - x) * tensor.log(1 - z)
        return sum([ce(inp, rec).sum(axis=1).mean() for inp, rec in pairs])

##################################################
def get(str):
    """ Evaluate str into a cost object, if it exists """
    obj = globals()[str]
    if issubclass(obj, Cost):
        return obj
    else:
        raise NameError(str)
