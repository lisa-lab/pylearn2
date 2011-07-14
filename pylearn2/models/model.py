from theano import tensor as T

class Model(object):
    def score(self, V):
        return T.grad(- self.free_energy(V).sum(), V)
    #

    def censor_updates(self, updates):
        pass
    #

    def free_energy(self, V):
        raise NotImplementedError()
    #

    def get_params(self):
        raise NotImplementedError()
    #

