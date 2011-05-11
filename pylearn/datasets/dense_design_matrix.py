import numpy as N
import copy

class DenseDesignMatrix(object):

    def __init__(self, X, y = None, view_converter = None, rng = None):
        self.X = X
        self.y = y
        self.view_converter = view_converter
        if rng is None:
            rng = N.random.RandomState([17,2,946])
        #
        self.default_rng = copy.copy(rng)
        self.rng = rng
        self.compress = False
        self.design_loc = None
    #

    def use_design_loc(self, path):
        self.design_loc = path
    #

    def enable_compression(self):
        self.compress = True
    #

    def __getstate__(self):
        rval = copy.copy(self.__dict__)

        if self.compress:
            rval['compress_min'] = rval['X'].min(axis=0)
            rval['X'] -= rval['compress_min']
            rval['compress_max'] = rval['X'].max(axis=0)
            rval['compress_max'][rval['compress_max']==0] = 1
            rval['X'] *= 255. / rval['compress_max']
            rval['X'] = N.cast['uint8'](rval['X'])

        if self.design_loc is not None:
            N.save(self.design_loc, rval['X'])
            del rval['X']

        return rval

    def __setstate__(self, d):

        if d['design_loc'] is not None:
            d['X'] = N.load(d['design_loc'])

        if d['compress']:
            X = d['X']
            mx = d['compress_max']
            mn = d['compress_min']
            del d['compress_max']
            del d['compress_min']
            d['X'] = 0
            self.__dict__.update(d)
            self.X = N.cast['float32'](X)*mx/255.+mn
        else:
            self.__dict__.update(d)


    def get_stream_position(self):
        return copy.copy(self.rng)

    def set_stream_position(self, pos):
        self.rng = copy.copy(pos)

    def restart_stream(self):
        self.reset_RNG()

    def reset_RNG(self):
        if 'default_rng' not in dir(self):
            self.default_rng = N.random.RandomState([17,2,946])
        self.rng = copy.copy(self.default_rng)
    #

    def apply_preprocessor(self, preprocessor, can_fit = False):
        preprocessor.apply(self, can_fit)
    #

    def get_topological_view(self, mat = None):
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset that has no view converter")
        #

        if mat is None:
            mat = self.X

        return self.view_converter.design_mat_to_topo_view(mat)
    #

    def get_weights_view(self, mat):
        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset that has no view converter")
        #

        return self.view_converter.design_mat_to_weights_view(mat)


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

    def view_shape(self):
        return self.view_converter.view_shape()
#

class DefaultViewConverter:
    def __init__(self, shape):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        #
    #

    def view_shape(self):
        return self.shape

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

    def design_mat_to_weights_view(self, X):
        return self.design_mat_to_topo_view(X)
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
