"""
Definitions of the cost for the gated-autoencoder.
"""

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.space import VectorSpace


class SymmetricCost(DefaultDataSpecsMixin, Cost):
    """
    Summary (Class representing the symmetric cost).

    Subclasses can define the type of data they will use.
    Mean reconstruction error is used for real valued data
    and cross-Entropy loss is used for binary.

    See Also
    --------
    "Gradient-based learning of higher-order image features"
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

        Returns
        -------
        Cost: theano_like expression
            Representation of the cost
        """
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):
        """
        Returns a theano expression for the cost function.

        Returns a symbolic expression for a cost function applied to the
        minibatch of data.
        Optionally, may return None. This represents that the cost function
        is intractable but may be optimized via the get_gradients method.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments. Not used by the base class.
        """
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
    Summary (Symmetric cost for real valued data).

    See Also
    --------
    "Gradient-based learning of higher-order image features"
    """
    @staticmethod
    def cost(x, y, rx, ry):
        """
        Summary (Definition of the cost).

        Mean squared reconstruction error.

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

        Returns
        -------
        Cost: theano_like expression
            Representation of the cost

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
    Summary (Normalized Symmetric cost for real valued data).

    Notes
    -----
    Value used to observe the percentage of reconstruction.
    """
    @staticmethod
    def cost(x, y, rx, ry):
        """
        Summary (Definition of the cost).

        Normalized Mean squared reconstruction error. Values
        between 0 and 1.

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

        Returns
        -------
        Cost: theano_like expression
            Representation of the cost

        Notes
        -----
        Do not use this function to train, only to monitor the
        average percentage of reconstruction achieved when training on
        real valued data.
        """
        num = (((0.5*((x - rx)**2)) + (0.5*((y - ry)**2)))).sum(axis=1).mean()
        den = ((0.5*(x.norm(2, 1)**2)) + (0.5*(y.norm(2, 1)**2))).mean()
        return num/den
