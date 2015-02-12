from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import wraps


class SymmetricCost(DefaultDataSpecsMixin, Cost):
    @staticmethod
    def cost(X, Y, rX, rY):
        """
        Symmetric reconstruction cost.

        Parameters
        ----------
        X : tensor_like
            Theano symbolic representing the first input minibatch.
            Assumed to be 2-tensors, with the first dimension
            indexing training examples and the second indexing
            data dimensions.
        Y : tensor_like
            Theano symbolic representing the seconde input minibatch.
            Assumed to be 2-tensors, with the first dimension
            indexing training examples and the second indexing
            data dimensions.
        rX : tensor_like
            Reconstruction of the first minibatch achieved by the model.
        rY: tensor_like
            Reconstruction of the second minibatch achieved by the model.
        """
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):

        self.get_data_specs(model)[0].validate(data)
        X = data[:, :model.nvisX]
        Y = data[:, model.nvisX:]
        rX, rY = model.reconstructXY(data)
        return self.cost(X, Y, rX, rY)


class SymmetricMSRE(SymmetricCost):
    @staticmethod
    @wraps(SymmetricCost.cost)
    def cost(X, Y, rX, rY):
        """
        Notes
        -----
        Symmetric reconstruction cost as defined by Memisevic in:
        "Gradient-based learning of higher-order image features".
        This function only works with real valued data.
        """
        return (
            ((0.5*((X - rX)**2)) + (0.5*((Y - rY)**2)))).sum(axis=1).mean()


class NormalizedSymmetricMSRE(SymmetricCost):
    @staticmethod
    @wraps(SymmetricCost.cost)
    def cost(X, Y, rX, rY):
        """
        Notes
        -----
        Do not use this function to train, only to monitor the
        average percentage of reconstruction achieved when training on
        real valued data.
        """
        num = (((0.5*((X - rX)**2)) + (0.5*((Y - rY)**2)))).sum(axis=1).mean()
        den = ((0.5*(X.norm(2, 1)**2)) + (0.5*(Y.norm(2, 1)**2))).mean()
        return num/den
