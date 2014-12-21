from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace
import theano


class SymmetricCost(DefaultDataSpecsMixin, Cost):
    """
    Class representing the symmetric cost, subclasses can
    define the type of data they will use
    real -> Mean Reconstruction error
    binary -> Cross-Entropy loss
    """
    @staticmethod
    def cost(x, y, rx, ry):
        """
        Symmetric reconstruction cost.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the first input minibatch.
            Assumed to be 2-tensors, with the first dimension
            indexing training examples and the second indexing
            data dimensions.
        y : tensor_like
            Theano symbolic representing the seconde input minibatch.
            Assumed to be 2-tensors, with the first dimension
            indexing training examples and the second indexing
            data dimensions.
        rx : tensor_like
            Reconstruction of the first minibatch by the model.
        ry: tensor_like
            Reconstruction of the second minibatch by the model.
        """
        raise NotImplementedError

    @wraps(Cost.expr)
    def expr(self, model, data, *args, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        x, y = data
        input_space = model.get_input_space()
        if not isinstance(input_space.components[0], VectorSpace):
            conv = input_space.components[0]
            vec = VectorSpace(conv.get_total_dimension())
            x = conv.format_as(x, vec)
        if not isinstance(input_space.components[1], VectorSpace):
            conv = input_space.components[1]
            vec = VectorSpace(conv.get_total_dimension())
            y = conv.format_as(y, vec)
        rx, ry = model.reconstructXY((x, y))
        return self.cost(x, y, rx, ry)


class SymmetricMSRE(SymmetricCost):
    """
    Symmetric cost for real valued data.
    """
    @staticmethod
    @wraps(SymmetricCost.cost)
    def cost(x, y, rx, ry):
        """
        Notes
        -----
        Symmetric reconstruction cost as defined by Memisevic in:
        "Gradient-based learning of higher-order image features".
        This function only works with real valued data.
        """
        return (
            ((0.5*((x - rx)**2)) + (0.5*((y - ry)**2)))).sum(axis=1).mean()


class NormalizedSymmetricMSRE(SymmetricCost):
    """
    Normalized Symmetric cost for real valued data.
    Values between one and zero.
    """
    @staticmethod
    @wraps(SymmetricCost.cost)
    def cost(x, y, rx, ry):
        """
        Notes
        -----
        Do not use this function to train, only to monitor the
        average percentage of reconstruction achieved when training on
        real valued data.
        """
        num = (((0.5*((x - rx)**2)) + (0.5*((y - ry)**2)))).sum(axis=1).mean()
        den = ((0.5*(x.norm(2, 1)**2)) + (0.5*(y.norm(2, 1)**2))).mean()
        return num/den
