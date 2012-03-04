import warnings
import copy
import numpy as np
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

class ExtractGridPatches(object):
    """ Converts a dataset into a dataset of patches
        extracted along a regular grid from each image.
        The order of the images is preserved.
    """
    def __init__(self, patch_shape, patch_stride):
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride

    def apply(self, dataset, can_fit = False):

        X = dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("""ExtractGridPatches with """+str(len(self.patch_shape))
                +""" topological dimensions called on dataset with """+
                str(num_topological_dimensions)+""".""")

        num_patches = X.shape[0]

        max_strides = [X.shape[0]-1]

        for i in xrange(num_topological_dimensions):
            patch_width = self.patch_shape[i]
            data_width = X.shape[i+1]
            last_valid_coord = data_width - patch_width
            if last_valid_coord < 0:
                raise ValueError('On topological dimension '+str(i)+\
                        ', the data has width '+str(data_width)+' but the '+\
                        'requested patch width is '+str(patch_width))
            stride = self.patch_stride[i]
            if stride == 0:
                max_stride_this_axis = 0
            else:
                max_stride_this_axis = last_valid_coord / stride

            num_strides_this_axis = max_stride_this_axis + 1

            max_strides.append(max_stride_this_axis)

            num_patches *= num_strides_this_axis

        #batch size
        output_shape = [ num_patches ]
        #topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        #number of channels
        output_shape.append(X.shape[-1])

        output = np.zeros(output_shape, dtype = X.dtype)

        channel_slice = slice(0,X.shape[-1])

        coords = [ 0 ]  *  (num_topological_dimensions + 1)

        keep_going = True
        i = 0
        while keep_going:

            args = [ coords[0] ]

            for j in xrange(num_topological_dimensions):
                coord = coords[j+1] * self.patch_stride[j]
                args.append(slice(coord,coord+self.patch_shape[j]))
            #end for j

            args.append(channel_slice)

            patch = X[args]
            output[i,:] = patch
            i += 1

            #increment coordinates
            j = 0

            keep_going = False

            while not keep_going:
                if coords[-(j+1)] < max_strides[-(j+1)]:
                    coords[-(j+1)] += 1
                    keep_going = True
                else:
                    coords[-(j+1)] = 0

                    if j == num_topological_dimensions:
                        break

                    j = j + 1
                    #end if j
                #end if coords
            #end while not continue
        #end while continue

        dataset.set_topological_view(output)

class ReassembleGridPatches(object):
    """ Converts a dataset of patches into a dataset of full examples
        This is the inverse of ExtractGridPatches for patch_stride=patch_shape
    """
    def __init__(self, orig_shape, patch_shape):
        self.patch_shape = patch_shape
        self.orig_shape = orig_shape

    def apply(self, dataset, can_fit = False):

        patches = dataset.get_topological_view()

        num_topological_dimensions = len(patches.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("""ReassembleGridPatches with """+str(len(self.patch_shape))
                +""" topological dimensions called on dataset with """+
                str(num_topological_dimensions)+""".""")

        num_patches = patches.shape[0]

        num_examples = num_patches

        for im_dim, patch_dim in zip(self.orig_shape, self.patch_shape):

            if im_dim % patch_dim != 0:
                raise Exception('Trying to assemble patches of shape '+\
                        str(self.patch_shape)+' into images of shape '+\
                        str(self.orig_shape))

            patches_this_dim = im_dim / patch_dim

            if num_examples % patches_this_dim != 0:
                raise Exception('Trying to re-assemble '+str(num_patches) + \
                        ' patches of shape '+str(self.patch_shape)+\
                        ' into images of shape '+str(self.orig_shape))
            num_examples /= patches_this_dim

        #batch size
        reassembled_shape = [ num_examples ]
        #topological dimensions
        for dim in self.orig_shape:
            reassembled_shape.append(dim)
        #number of channels
        reassembled_shape.append(patches.shape[-1])

        reassembled = np.zeros(reassembled_shape, dtype = patches.dtype)

        channel_slice = slice(0,patches.shape[-1])

        coords = [ 0 ]  *  (num_topological_dimensions + 1)


        max_strides = [ num_examples - 1]
        for dim, pd in zip(self.orig_shape, self.patch_shape):
            assert dim % pd == 0
            max_strides.append(dim/pd-1)

        keep_going = True
        i = 0
        while keep_going:

            args = [ coords[0] ]

            for j in xrange(num_topological_dimensions):
                coord = coords[j+1]
                args.append(slice(coord*self.patch_shape[j],(coord+1)*self.patch_shape[j]))
                assert (coord + 1) * self.patch_shape[j] <= reassembled.shape[j+1]
            #end for j

            args.append(channel_slice)

            try:
                patch = patches[i,:]
            except IndexError:
                raise IndexError('Gave index of '+str(i)+',: into thing of shape '+str(patches.shape))

            reassembled[args] = patch
            i += 1

            #increment coordinates
            j = 0

            keep_going = False

            while not keep_going:
                if coords[-(j+1)] < max_strides[-(j+1)]:
                    coords[-(j+1)] += 1
                    keep_going = True
                else:
                    coords[-(j+1)] = 0

                    if j == num_topological_dimensions:
                        break

                    j = j + 1
                    #end if j
                #end if coords
            #end while not continue
        #end while continue

        dataset.set_topological_view(reassembled)

class ExtractPatches(object):
    """ Converts an image dataset into a dataset of patches
        extracted at random from the original dataset. """
    def __init__(self, patch_shape, num_patches, rng = None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1,2,3])
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

        output = np.zeros(output_shape, dtype = X.dtype)

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

class MakeUnitNorm(object):
    def __init__(self):
        pass

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X_norm = np.sqrt(np.sum(X**2, axis=1))
        X /= X_norm[:,None]
        dataset.set_design_matrix(X)

class RemoveMean(object):
    def __init__(self, axis=0):
        self.axis=axis

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X -= X.mean(axis=self.axis)
        dataset.set_design_matrix(X)

class Standardize(object):

    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4):
        self.global_mean= global_mean
        self.global_std = global_std
        self.std_eps = std_eps

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()

        # remove mean across all dataset, or along each dimension
        mean= np.mean(X) if self.global_mean else np.mean(X, axis=0)
        # divide by std across all dataset, or along each dimension
        std = np.std(X)  if self.global_std  else np.std(X, axis=0)

        dataset.set_design_matrix( (X - mean) / (self.std_eps + std) )


class RemapInterval(object):
    def __init__(self, map_from, map_to):
        assert map_from[0] < map_from[1] and len(map_from) == 2
        assert map_to[0] < map_to[1] and len(map_to) == 2
        self.map_from = [np.float(x) for x in map_from]
        self.map_to   = [np.float(x) for x in map_to]

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X = (X - self.map_from[0]) / np.diff(self.map_from)
        X = X * np.diff(self.map_to) + self.map_to[0]
        dataset.set_design_matrix(X)

class PCA_ViewConverter(object):
    def __init__(self, to_pca, to_input, to_weights, orig_view_converter):
        self.to_pca = to_pca
        self.to_input = to_input
        self.to_weights = to_weights
        if orig_view_converter is None:
            raise ValueError("It doesn't make any sense to make a PCA view converter when there's no original view converter to define a topology in the first place")
        self.orig_view_converter = orig_view_converter

    def view_shape(self):
        return self.orig_view_converter.shape

    def design_mat_to_topo_view(self, X):
        return self.orig_view_converter.design_mat_to_topo_view(self.to_input(X))

    def design_mat_to_weights_view(self, X):
        return self.orig_view_converter.design_mat_to_weights_view(self.to_weights(X))

    def topo_view_to_design_mat(self, V):
        return self.to_pca(self.orig_view_converter.topo_view_to_design_mat(V))



class PCA(object):
    def __init__(self, num_components):
        self.num_components = num_components
        self.pca = None
        self.input = T.matrix()
        self.output = T.matrix()

    def apply(self, dataset, can_fit = False):
        if self.pca is None:
            assert can_fit
            from pylearn2 import pca
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

class Downsample(object):
    def __init__(self, sampling_factor):
        """
            downsamples the topological view

            parameters
            ----------
            sampling_factor: a list or array with one element for
                            each topological dimension of the data
        """

        self.sampling_factor = sampling_factor

    def apply(self, dataset, can_fit = False):
        X = dataset.get_topological_view()

        d = len(X.shape) - 2

        assert d in [2,3]
        assert X.dtype == 'float32' or X.dtype == 'float64'

        if d == 2:
            X = X.reshape([ X.shape[0], X.shape[1], X.shape[2], 1, X.shape[3] ])

        kernel_size = 1

        kernel_shape = [  X.shape[-1] ]

        for factor in self.sampling_factor:
            kernel_size *= factor
            kernel_shape.append(factor)


        if d == 2:
            kernel_shape.append(1)

        kernel_shape.append(X.shape[-1])

        kernel_value = 1. / float(kernel_size)

        kernel = np.zeros(kernel_shape, dtype=X.dtype)

        for i in xrange(X.shape[-1]):
            kernel[i,:,:,:,i] = kernel_value

        from theano.tensor.nnet.Conv3D import conv3D

        X_var = T.TensorType( broadcastable = [ s == 1 for s in X.shape],
                            dtype = X.dtype)()

        downsampled = conv3D(X_var, kernel, np.zeros(X.shape[-1],X.dtype), kernel_shape[1:-1])

        f = function([X_var], downsampled)

        X = f(X)

        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], X.shape[4]])

        dataset.set_topological_view(X)

class GlobalContrastNormalization(object):
    def __init__(self, subtract_mean = True, std_bias = 10.0, use_norm = False):
        """

        Optionally subtracts the mean of each example
        Then divides each example either by the standard deviation of the pixels
        contained in that example or by the norm of that example

        Parameters:

            subtract_mean: boolean, if True subtract the mean of each example
            std_bias: Add this amount inside the square root when computing
                      the standard deviation or the norm
            use_norm: If True uses the norm instead of the standard deviation


            The default parameters of subtract_mean = True, std_bias = 10.0,
            use_norm = False are used in replicating one step of the preprocessing
            used by Coates, Lee and Ng on CIFAR10 in their paper "An Analysis
            of Single Layer Networks in Unsupervised Feature Learning"


        """

        self.subtract_mean = subtract_mean
        self.std_bias = std_bias
        self.use_norm = use_norm

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()

        assert X.dtype == 'float32' or X.dtype == 'float64'

        if self.subtract_mean:
            X -= X.mean(axis=1)[:,None]

        if self.use_norm:
            scale = np.sqrt( np.square(X).sum(axis=1) + self.std_bias)
        else:
            #use standard deviation
            scale = np.sqrt( np.square(X).mean(axis=1) + self.std_bias)

        eps = 1e-8
        scale[scale < eps] = 1.

        X /= scale[:,None]

        dataset.set_design_matrix(X)



class ZCA(object):
    def __init__(self, n_components=None, n_drop_components=None, filter_bias=0.1):
        warnings.warn("""This ZCA preprocessor class is known to yield very different results on different platforms. If you plan to conduct experiments with this preprocessing on multiple machines, it is probably a good idea to do the preprocessing on a single machine and copy the preprocessed datasets to the others, rather than preprocessing the data independently in each location.""")
        #TODO: test to see if differences across platforms
        # e.g., preprocessing STL-10 patches in LISA lab versus on
        # Ian's Ubuntu 11.04 machine
        # are due to the problem having a bad condition number or due to
        # different version numbers of scipy or something
        self.n_components = n_components
        self.n_drop_components =n_drop_components
        self.copy = True
        self.filter_bias = filter_bias
        self.has_fit_ = False

    def fit(self, X):
        assert X.dtype in ['float32','float64']
        assert not np.any(np.isnan(X))

        assert len(X.shape) == 2

        n_samples = X.shape[0]

        if self.copy:
            X = X.copy()

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        print 'computing zca'
        eigs, eigv = linalg.eigh(np.dot(X.T, X)/X.shape[0])

        assert not np.any(np.isnan(eigs))
        assert not np.any(np.isnan(eigv))

        if self.n_components:
            eigs = eigs[:self.n_components]
            eigv = eigv[:,:self.n_components]
        #
        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:,self.n_drop_components:]
        #

        self.P_ = np.dot(
                eigv * np.sqrt(1.0/(eigs+self.filter_bias)),
                eigv.T)


        #print 'zca components'
        #print np.square(self.P_).sum(axis=0)



        assert not np.any(np.isnan(self.P_))

        self.has_fit_ = True
    #

    def apply(self, dataset, can_fit = False):
        X = dataset.get_design_matrix()
        assert X.dtype in ['float32','float64']

        if not self.has_fit_:
            assert can_fit
            self.fit(X)
        #

        new_X =  np.dot(X-self.mean_, self.P_)

        #print 'mean absolute difference between new and old X'+str(np.abs(X-new_X).mean())

        dataset.set_design_matrix(new_X)
    #
#






