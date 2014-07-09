from theano import tensor
from theano.compat.python2x import OrderedDict

from pylearn2.model_extensions.model_extension import ModelExtension
from pylearn2.utils import is_iterable
from pylearn2.utils import py_number_types


class GradientClipping(ModelExtension):
    def __init__(self, clipping_value, exclude_params=None):
        """
        Clip the gradient if the norm is above clipping_value.

        Parameters
        ----------
        clipping_value : float or int
            The squared norm above which to clip the gradient.
        exclude_params : list of strings
            The names of the parameters that are excluded from clipping
        """
        assert isinstance(clipping_value, py_number_types)
        if exclude_params is not None:
            assert is_iterable(exclude_params)
            assert all(isinstance(param_name, basestring)
                       for param_name in exclude_params)
        self.__dict__.update(locals())
        del self.self

    def post_modify_updates(self, updates):
        # TODO Deal with infinite/nan sq_norm, in which case we rescale
        # the gradient to something small
        sq_norm = tensor.sum(tensor.sum(param_gradient ** 2)
                             for param, param_gradient in updates.iteritems()
                             if param.name not in self.exclude_params)

        clipped_updates = OrderedDict()
        for param, param_gradient in updates.iteritems():
            if param.name not in self.exclude_params:
                clipped_updates[param] = tensor.switch(
                    tensor.ge(sq_norm, self.clipping_value),
                    param_gradient / sq_norm * self.clipping_value,
                    param_gradient
                )
        updates.update(clipped_updates)
        return updates
