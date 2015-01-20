"""
Implements gradient clipping as a cost wrapper.
"""
from functools import wraps

from theano.compat import six
from theano import tensor

from pylearn2.compat import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.utils import is_iterable
from pylearn2.utils import py_number_types


class GradientClipping(object):
    """
    Clip the gradient if the sum of squared gradient norms
    is above the clipping_value.

    Parameters
    ----------
    clipping_value : float or int
        The norm above which to clip the gradient.
    cost : Cost object
        The actual cost to use for this model.
    exclude_params : list of strings, optional
        The names of the parameters that are excluded from clipping
    """
    def __init__(self, clipping_value, cost, exclude_params=[]):
        assert isinstance(clipping_value, py_number_types)
        assert isinstance(cost, Cost)
        assert is_iterable(exclude_params)
        assert all(isinstance(param_name, six.string_types)
                   for param_name in exclude_params)
        self.__dict__.update(locals())
        del self.self

    def __getattr__(self, attr):
        # This effectively makes this class a subclass
        # of the class instance that was passed in the constructor
        return getattr(self.cost, attr)

    @wraps(Cost.get_gradients)
    def get_gradients(self, model, data, **kwargs):
        gradients, updates = self.cost.get_gradients(model, data, **kwargs)

        norm = tensor.sqrt(tensor.sum(
            [tensor.sum(param_gradient ** 2) for param, param_gradient
             in six.iteritems(gradients)
             if param.name not in self.exclude_params]
        ))

        clipped_gradients = OrderedDict()
        for param, param_gradient in six.iteritems(gradients):
            if param.name not in self.exclude_params:
                clipped_gradients[param] = tensor.switch(
                    tensor.ge(norm, self.clipping_value),
                    param_gradient / norm * self.clipping_value,
                    param_gradient
                )
        gradients.update(clipped_gradients)
        return gradients, updates
