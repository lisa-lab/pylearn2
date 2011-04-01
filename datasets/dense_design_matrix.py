import numpy as N

class DenseDesignMatrix(object):
    def __init__(self, X, y = None, view_converter = None, rng = None):
        self.X = X
        self.y = y
        self.view_converter = view_converter
        if rng is None:
            rng = N.random.RandomState([17,2,946])
        #
        self.rng = rng
    #

    def apply_preprocessor(self, preprocessor, can_fit = False):
        preprocessor.apply(self, can_fit)
    #

    def get_topological_view(self):
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset that has no view converter")
        #

        return self.view_converter.design_mat_to_topo_view(self.X)
    #

    def set_topological_view(self, V):
        assert not N.any(N.isnan(V))
        self.view_converter = DefaultViewConverter(V.shape[1:])
        self.X = self.view_converter.topo_view_to_design_mat(V)
        assert not N.any(N.isnan(self.X))
    #

    def get_design_matrix(self):
        return self.X
    #

    def set_design_matrix(self, X):
        assert not N.any(N.isnan(X))
        self.X = X
    #

    def get_batch_design(self, batch_size):
        idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        return self.X[idx:idx+batch_size,:]
    #

    def get_batch_topo(self, batch_size):
        return self.view_converter.design_mat_to_topo_view(self.get_batch_design(batch_size))
    #
#

class DefaultViewConverter:
    def __init__(self, shape):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        #
    #

    def design_mat_to_topo_view(self, X):
        batch_size = X.shape[0]

        channel_shape = [ batch_size ]
        for dim in self.shape[:-1]:
            channel_shape.append(dim)
        channel_shape.append(1)

        channels = [
                    X[:,i*self.pixels_per_channel:(i+1)*self.pixels_per_channel].reshape(*channel_shape)
                    for i in xrange(self.shape[-1])
                    ]

        rval = N.concatenate(channels,axis=len(self.shape))

        assert len(rval.shape) == len(self.shape) + 1

        return rval
    #

    def topo_view_to_design_mat(self, V):
        if N.any( N.asarray(self.shape) != N.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size followed by '
                            +str(shape)+' given tensor of shape '+str(V.shape))
        #

        batch_size = V.shape[0]

        channels = [
                    V[:,:,:,i].reshape(batch_size,self.pixels_per_channel)
                    for i in xrange(3)
                    ]

        return N.concatenate(channels,axis=1)
    #
#
