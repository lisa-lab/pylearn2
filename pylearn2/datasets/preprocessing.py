import copy
import numpy as N
from scipy import linalg
from theano import function
import theano.tensor as T

class Pipeline(object):
    def __init__(self):
        self.items = []
    #

    def apply(self, dataset, can_fit = False):
        for item in self.items:
            item.apply(dataset, can_fit)
        #
    #
#


class ExtractPatches(object):
    def __init__(self, patch_shape, num_patches, rng = None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = N.random.RandomState([1,2,3])
        #
    #

    def apply(self, dataset, can_fit = False):
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("""ExtractPatches with """+str(len(self.patch_shape))
                +""" topological dimensions called on dataset with """+
                str(num_topological_dimensions)+""".""")

        #batch size
        output_shape = [ self.num_patches ]
        #topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        #number of channels
        output_shape.append(X.shape[-1])

        output = N.zeros(output_shape, dtype = X.dtype)

        channel_slice = slice(0,X.shape[-1])

        for i in xrange(self.num_patches):
            args = []

            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j+1]-self.patch_shape[j]
                coord = rng.randint(max_coord+1)
                args.append(slice(coord,coord+self.patch_shape[j]))
            #

            args.append(channel_slice)

            output[i,:] = X[args]
        #

        dataset.set_topological_view(output)
    #


class RemoveMean(object):
    def __init__(self):
        pass

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X -= X.mean(axis=0)
        dataset.set_design_matrix(X)


class PCA_ViewConverter:
    def __init__(self, to_pca, to_input, to_weights, orig_view_converter):
        self.to_pca = to_pca
        self.to_input = to_input
        self.to_weights = to_weights
        if orig_view_converter is None:
            raise ValueError("It doesn't make any sense to make a PCA view converter when there's no original view converter to define a topology in the first place")
        self.orig_view_converter = orig_view_converter
    #

    def view_shape(self):
        return self.orig_view_converter.shape

    def design_mat_to_topo_view(self, X):
        return self.orig_view_converter.design_mat_to_topo_view(self.to_input(X))
    #

    def design_mat_to_weights_view(self, X):
        return self.orig_view_converter.design_mat_to_weights_view(self.to_weights(X))
    #

    def topo_view_to_design_mat(self, V):
        return self.to_pca(self.orig_view_converter.topo_view_to_design_mat(X))
    #
#



class PCA(object):
    def __init__(self, num_components):
        self.num_components = num_components
        self.pca = None
        self.input = T.matrix()
        self.output = T.matrix()

    def apply(self, dataset, can_fit = False):
        if self.pca is None:
            assert can_fit
            from framework import pca
            self.pca = pca.CovEigPCA(self.num_components)
            self.pca.train(dataset.get_design_matrix())

            self.transform_func = function([self.input],self.pca(self.input))
            self.invert_func = function([self.output],self.pca.reconstruct(self.output))
            self.convert_weights_func = function([self.output],self.pca.reconstruct(self.output,add_mean = False))
        #

        orig_data = dataset.get_design_matrix()#rm
        dataset.set_design_matrix(self.transform_func(dataset.get_design_matrix()))
        proc_data = dataset.get_design_matrix()#rm
        orig_var = orig_data.var(axis=0)
        proc_var = proc_data.var(axis=0)
        assert proc_var[0] > orig_var.max()
        print 'original variance: '+str(orig_var.sum())
        print 'processed variance: '+str(proc_var.sum())

        dataset.view_converter = PCA_ViewConverter(self.transform_func,self.invert_func,self.convert_weights_func, dataset.view_converter)
    #
#



class GlobalContrastNormalization(object):
    def __init__(self, subtract_mean = True, std_bias = 10.0):
        self.subtract_mean = subtract_mean
        self.std_bias = std_bias

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()
        assert X.dtype == 'float32' or X.dtype == 'float64'

        if self.subtract_mean:
            X -= X.mean(axis=1)[:,None]
        std = N.sqrt( (X**2.).mean(axis=1) + self.std_bias)
        X /= std[:,None]

        dataset.set_design_matrix(X)



class ZCA(object):
    def __init__(self, n_components=None, n_drop_components=None, filter_bias=0.1):
        self.n_components = n_components
        self.n_drop_components =n_drop_components
        self.copy = True
        self.filter_bias = filter_bias
        self.has_fit_ = False

    def fit(self, X):
        assert X.dtype in ['float32','float64']
        assert not N.any(N.isnan(X))

        assert len(X.shape) == 2

        n_samples = X.shape[0]

        if self.copy:
            X = X.copy()

        # Center data
        self.mean_ = N.mean(X, axis=0)
        X -= self.mean_

        print 'computing zca'
        eigs, eigv = linalg.eigh(N.dot(X.T, X)/X.shape[0])

        assert not N.any(N.isnan(eigs))
        assert not N.any(N.isnan(eigv))

        if self.n_components:
            eigs = eigs[:self.n_components]
            eigv = eigv[:,:self.n_components]
        #
        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:,self.n_drop_components:]
        #

        self.P_ = N.dot(
                eigv * N.sqrt(1.0/(eigs+self.filter_bias)),
                eigv.T)


        print 'zca components'
        print N.square(self.P_).sum(axis=0)



        assert not N.any(N.isnan(self.P_))

        self.has_fit_ = True
    #

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()
        assert X.dtype in ['float32','float64']

        if not self.has_fit_:
            assert can_fit
            self.fit(X)
        #

        new_X =  N.dot(X-self.mean_, self.P_)

        print 'mean absolute difference between new and old X'+str(N.abs(X-new_X).mean())

        dataset.set_design_matrix(new_X)
    #
#






