import copy
import numpy as N
from scipy import linalg

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


"""
James says he doesn't actually use this in the ssRBM experiments
class RemoveColorShift(object):
    def __init(self):
        pass
    #

    def apply(self, dataset, can_fit = False):
        dataset.
"""

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

class GlobalContrastNormalization(object):
    def __init__(self, std_bias = 10.0):
        self.std_bias = std_bias

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()

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
        assert len(X.shape) == 2

        n_samples = X.shape[0]

        if self.copy:
            X = X.copy()

        # Center data
        self.mean_ = N.mean(X, axis=0)
        X -= self.mean_

        print 'computing zca'
        eigs, eigv = linalg.eigh(N.dot(X.T, X)/X.shape[0])

        if self.n_components:
            eigs = eigs[:self.n_components]
            eigv = eigv[:,:self.n_components]
        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:,self.n_drop_components:]
        self.P_ = N.dot(
                eigv * N.sqrt(1.0/(eigs+self.filter_bias)),
                eigv.T)

        self.has_fit_ = True

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()

        if not self.has_fit_:
            assert can_fit
            self.fit(X)

        X =  N.dot(X-self.mean_, self.P_)

        dataset.set_design_matrix(X)







