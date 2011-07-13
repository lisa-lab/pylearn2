import theano.tensor as T


class EnergyFunction(object):
    def __init__(self):
        pass
    #

    def score(self, X):
        assert X.dtype.find('int') == -1

        X_name = 'X' if X.name is None else X.name

        E = self.free_energy(X)

        #There should be one energy value for each example in the batch
        assert len(E.type.broadcastable) == 1

        dummy = T.sum(E)
        rval =  T.grad(dummy,X)

        rval.name = 'score('+X_name+')'

        return rval
    #

    def __call__(self, X):
        raise NotImplementedError(str(type(self))+' has not implemented __call__(self,X) which should return the energy of X')
    #
#
