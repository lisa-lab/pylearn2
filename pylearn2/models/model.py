from theano import tensor as T
import copy

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

    def redo_theano(self):
        pass
    #

    def get_input_dim(self):
        raise NotImplementedError()
    #

    def __getstate__(self):
        d = copy.copy(self.__dict__)

        #remove everything set up by redo_theano

        for name in self.names_to_del:
            if name in d:
                del d[name]

        return d
    #

    def __setstate__(self, d):
        self.__dict__.update(d)
    #

    def __init__(self):
        self.names_to_del = []
    #

    def register_names_to_del(self, names):
        for name in names:
            if name not in self.names_to_del:
                self.names_to_del.append(name)
            #
        #
    #
#
