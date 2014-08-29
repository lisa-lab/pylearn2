from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class SymmetricCost(DefaultDataSpecsMixin, Cost):
    @staticmethod
    def cost(target, output):
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):

        self.get_data_specs(model)[0].validate(data)
        X = data[:, :model.nvisX]
        Y = data[:, model.nvisY:]
        rX, rY = model.reconstructXY(data)
        return self.cost(X, Y, rX, rY)


class SymmetricMSRE(SymmetricCost):
    """
    Symmetric error as defined by Memisevic in:
    "Gradient-based learning of higher-order image features".
    """
    @staticmethod
    def cost(a, b, c, d):
        return (
            ((0.5*((a - c)**2)) + (0.5*((b - d)**2)))).sum(axis=1).mean()


class NormalizedSymmetricMSRE(SymmetricCost):
    """
    Do not use this function to train, only to monitor the
    average percentage of reconstruction achieved when training on
    real valued data.
    """
    @staticmethod
    def cost(a, b, c, d):
        num = (((0.5*((a - c)**2)) + (0.5*((b - d)**2)))).sum(axis=1).mean()
        den = ((0.5*(a.norm(2, 1)**2)) + (0.5*(b.norm(2, 1)**2))).mean()
        return num/den
