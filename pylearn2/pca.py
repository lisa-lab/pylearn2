# Standard library imports
import sys

# Third-party imports
import numpy
N = numpy
import warnings
from scipy import linalg, sparse
# Warning: ridiculous.
try:
    # scipy 0.9
    from scipy.sparse.linalg import eigsh as eigen_symmetric
except ImportError:
    try:
        # scipy 0.8
        from scipy.sparse.linalg import eigen_symmetric
    except ImportError:
        try:
            # scipy 0.7
            from scipy.sparse.linalg.eigen.arpack import eigen_symmetric
        except ImportError:
            warnings.warn('Cannot import any kind of symmetric eigen' \
                ' decomposition function from scipy.sparse.linalg')
            return
from scipy.sparse.csr import csr_matrix
import theano
from theano import tensor
from theano.sparse import SparseType, structured_dot
from scipy import linalg
from scipy.sparse.csr import csr_matrix

try:
    from scipy.sparse.linalg import eigen_symmetric
except ImportError:
    #this was renamed to eigsh in scipy 0.9
    try:
        from scipy.sparse.linalg import eigsh as eigen_symmetric
    except ImportError:
        # TODO: change this to use warn()
        print ("couldn't import eigsh / eigen_symmetric from "
               "scipy.linalg.sparse, some of your pca functions "
               "may randomly fail later")
        print ("the fact that somebody is using this doesn't bode well "
               " since it's unlikely that the covariance matrix is sparse")


# Local imports
from pylearn2.base import Block
from pylearn2.utils import sharedX


class _PCABase(Block):
    """
    Block which transforms its input via Principal Component Analysis.

    This class is not intended to be instantiated directly. Use a
    subclass to select a particular PCA implementation.
    """

    def __init__(self, num_components=None, min_variance=0.0, whiten=False):
        """
        :type num_components: int
        :param num_components: this many components will be preserved, in
            decreasing order of variance (default None keeps all)

        :type min_variance: float
        :param min_variance: components with normalized variance [0-1] below
            this threshold will be discarded

        :type whiten: bool
        :param whiten: whether or not to divide projected features by their
            standard deviation
        """

        super(_PCABase, self).__init__()

        self.num_components = num_components
        self.min_variance = min_variance
        self.whiten = whiten

        self.W = None
        self.v = None
        self.mean = None

        self.component_cutoff = theano.shared(
                                    theano._asarray(0, dtype='int64'),
                                    name='component_cutoff')

        # This module really has no adjustable parameters -- once train()
        # is called once, they are frozen, and are not modified via gradient
        # descent.
        self._params = []

    def train(self, X, mean=None):
        """
        Compute the PCA transformation matrix.

        Given a rectangular matrix X = USV such that S is a diagonal matrix
        with X's singular values along its diagonal, returns W = V^-1.

        If mean is provided, X will not be centered first.

        :type X: numpy.ndarray, shape (n, d)
        :param X: matrix on which to train PCA

        :type mean: numpy.ndarray, shape (d)
        :param mean: feature means
        """

        if self.num_components is None:
            self.num_components = X.shape[1]

        # Center each feature.
        if mean is None:
            mean = X.mean(axis=0)
            X = X - mean

        # Compute eigen{values,vectors} of the covariance matrix.
        v, W = self._cov_eigen(X)

        # Build Theano shared variables
        # For the moment, I do not use borrow=True because W and v are
        # subtensors, and I want the original memory to be freed
        self.W = sharedX(W, name='W')
        self.v = sharedX(v, name='v')
        self.mean = sharedX(mean, name='mean')

        # Filter out unwanted components, permanently.
        self._update_cutoff()
        component_cutoff = self.component_cutoff.get_value(borrow=True)
        self.v.set_value(self.v.get_value(borrow=True)[:component_cutoff])
        self.W.set_value(self.W.get_value(borrow=True)[:, :component_cutoff])

    def __call__(self, inputs):
        """
        Compute and return the PCA transformation of the current data.

        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix on which to compute PCA
        """

        # Update component cutoff, in case min_variance or num_components has
        # changed (or both).

        #TODO: Looks like the person who wrote this function didn't know what
        #      they were doing
        # component_cutoff is a shared variable, so updating its value here has
        # NO EFFECT on the symbolic expression returned by this call (and what
        # this expression evalutes to can be modified by subsequent calls to
        # _update_cutoff)
        self._update_cutoff()

        normalized_mean = inputs - self.mean
        normalized_mean.name = 'normalized_mean'

        W = self.W[:, :self.component_cutoff]

        #TODO: this is inefficient, should make another shared variable where
        # this proprocessing is already done
        if self.whiten:
            W = W / tensor.sqrt(self.v[:self.component_cutoff])

        Y = tensor.dot(normalized_mean, W)

        return Y

    def get_weights(self):
        """
        Compute and return the matrix one should multiply with to get the
        PCA/whitened data
        """

        self._update_cutoff()

        component_cutoff = self.component_cutoff.get_value()

        W = self.W.get_value(borrow=False)
        W = W[:, :component_cutoff]

        if self.whiten:
            W /= N.sqrt(self.v.get_value(borrow=False)[:component_cutoff])

        return W

    def reconstruct(self, inputs, add_mean=True):
        """
        Given a PCA transformation of the current data, compute and return
        the reconstruction of the original input """
        self._update_cutoff()
        if self.whiten:
            inputs *= tensor.sqrt(self.v[:self.component_cutoff])
        X = tensor.dot(inputs, self.W[:, :self.component_cutoff].T)
        if add_mean:
            X = X + self.mean
        return X

    def _update_cutoff(self):
        """
        Update component cutoff shared var, based on current parameters.
        """

        assert self.num_components is not None and self.num_components > 0, \
            'Number of components requested must be >= 1'

        v = self.v.get_value(borrow=True)
        var_mask = v / v.sum() > self.min_variance
        assert numpy.any(var_mask), \
            'No components exceed the given min. variance'
        var_cutoff = 1 + numpy.where(var_mask)[0].max()

        self.component_cutoff.set_value(min(var_cutoff, self.num_components))

    def _cov_eigen(self, X):
        """
        Compute and return eigen{values,vectors} of X's covariance matrix.

        Returns:
            all eigenvalues in decreasing order
            matrix containing corresponding eigenvectors in its columns
        """
        raise NotImplementedError('Not implemented in _PCABase. Use a subclass (and implement it there).')


class SparseMatPCA(_PCABase):
    """ Does PCA on sparse  matrices. Does not do online PCA.
        This is for the case where X - X.mean() does not fit
        in memory (because it's dense) but
        N.dot( (X-X.mean()).T, X-X.mean() ) does  """
    def __init__(self, batch_size=50, **kwargs):
        super(SparseMatPCA, self).__init__(**kwargs)
        self.minibatch_size = batch_size

    def get_input_type(self):
        return csr_matrix

    def _cov_eigen(self, X):
        n, d = X.shape

        cov = numpy.zeros((d, d))
        batch_size = self.minibatch_size

        for i in xrange(0, n, batch_size):
            print '\tprocessing example', str(i)
            end = min(n, i + batch_size)
            x = X[i:end, :].todense() - self.mean_
            assert x.shape[0] == end - i

            prod = numpy.dot(x.T, x)
            assert prod.shape == (d, d)

            cov += prod

        cov /= n

        print 'computing eigens'
        v, W = linalg.eigh(cov, eigvals=(d - self.num_components, d - 1))

        # The resulting components are in *ascending* order of eigenvalue, and
        # W contains eigenvectors in its *columns*, so we simply reverse both.
        v, W = v[::-1], W[:, ::-1]
        return v, W

    def train(self, X):
        """
        Compute the PCA transformation matrix.

        Given a rectangular matrix X = USV such that S is a diagonal matrix
        with X's singular values along its diagonal, computes and returns W =
        V^-1.
        """

        assert sparse.issparse(X)

        # Compute feature means.
        print 'computing mean'
        self.mean_ = numpy.asarray(X.mean(axis=0))[0, :]

        super(SparseMatPCA, self).train(X, mean=self.mean_)

    def __call__(self, inputs):

        self._update_cutoff()

        Y = structured_dot(inputs, self.W[:, :self.component_cutoff])
        Z = Y - tensor.dot(self.mean, self.W[:, :self.component_cutoff])

        #TODO-- this is inefficient, should work by modifying W not Z
        if self.whiten:
            Z /= tensor.sqrt(self.v[:self.component_cutoff])
        return Z

    def function(self, name=None):
        """ Returns a compiled theano function to compute a representation """
        inputs = SparseType('csr', dtype=theano.config.floatX)()
        return theano.function([inputs], self(inputs), name=name)


class OnlinePCA(_PCABase):
    """Online PCA implementation. Requires pylearn1."""

    def __init__(self, minibatch_size=500, **kwargs):
        super(OnlinePCA, self).__init__(**kwargs)
        self.minibatch_size = minibatch_size

    def _cov_eigen(self, X):
        """
        Perform online computation of covariance matrix eigen{values,vectors}.
        """

        num_components = min(self.num_components, X.shape[1])

        from pylearn.algorithms import pca_online_estimator
        pca_estimator = pca_online_estimator.PcaOnlineEstimator(X.shape[1],
            n_eigen=num_components,
            minibatch_size=self.minibatch_size,
            centering=False
        )

        print >> sys.stderr, '*' * 50
        for i in range(X.shape[0]):
            if (i + 1) % (X.shape[0] / 50) == 0:
                sys.stderr.write('|')  # suppresses newline/whitespace.
            pca_estimator.observe(X[i, :])
        print >> sys.stderr

        v, W = pca_estimator.getLeadingEigen()

        # The resulting components are in *ascending* order of eigenvalue,
        # and W contains eigenvectors in its *rows*, so we reverse both and
        # transpose W.
        return v[::-1], W.T[:, ::-1]


class Cov:
    """ A covariance estimator that computes the covariance in small batches
        instead of with one huge matrix multiply, in order to prevent memory
        problems. It's call method has the same functionality as numpy.cov """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X):
        X = X.T
        m, n = X.shape
        mean = X.mean(axis=0)
        rval = N.zeros((n, n))
        for i in xrange(0, m, self.batch_size):
            B = X[i:i + self.batch_size, :] - mean
            rval += N.dot(B.T, B)
        return rval / float(m - 1)


class CovEigPCA(_PCABase):
    def __init__(self, cov_batch_size=None, **kwargs):
        super(CovEigPCA, self).__init__(**kwargs)
        if cov_batch_size is not None:
            self.cov = Cov(cov_batch_size)
        else:
            self.cov = numpy.cov

    def _cov_eigen(self, X):
        """
        Perform direct computation of covariance matrix eigen{values,vectors}.
        """
        v, W = linalg.eigh(self.cov(X.T))
        # The resulting components are in *ascending* order of eigenvalue, and
        # W contains eigenvectors in its *columns*, so we simply reverse both.
        return v[::-1], W[:, ::-1]


class SVDPCA(_PCABase):
    def _cov_eigen(self, X):
        """
        Compute covariance matrix eigen{values,vectors} via Singular Value
        Decomposition (SVD).
        """
        U, s, Vh = linalg.svd(X, full_matrices=False)
        # Vh contains eigenvectors in its *rows*, thus we transpose it.
        # s contains X's singular values in *decreasing* order, thus (noting
        # that X's singular values are the sqrt of cov(X'X)'s eigenvalues), we
        # simply square it.
        return s ** 2, Vh.T


class SparsePCA(_PCABase):
    def train(self, X, mean=None):
        print >> sys.stderr, ('WARNING: You should probably be using '
                              'SparseMatPCA, unless your design matrix fits '
                              'in memory.')

        n, d = X.shape
        # Can't subtract a sparse vector from a sparse matrix, apparently,
        # so here I repeat the vector to construct a matrix.
        mean = X.mean(axis=0)
        mean_matrix = csr_matrix(mean.repeat(n).reshape((d, n))).T
        X = X - mean_matrix

        super(SparsePCA, self).train(X, mean=numpy.asarray(mean).squeeze())

    def _cov_eigen(self, X):
        """
        Perform direct computation of covariance matrix eigen{values,vectors},
        given a scipy.sparse matrix.
        """

        v, W = eigen_symmetric(X.T.dot(X) / X.shape[0], k=self.num_components)

        # The resulting components are in *ascending* order of eigenvalue, and
        # W contains eigenvectors in its *columns*, so we simply reverse both.
        return v[::-1], W[:, ::-1]

    def __call__(self, inputs):
        """
        Compute and return the PCA transformation of sparse data.

        Precondition: self.mean has been subtracted from inputs.  The reason
        for this is that, as far as I can tell, there is no way to subtract a
        vector from a sparse matrix without constructing an intermediary dense
        matrix, in theano; even the hack used in train() won't do, because
        there is no way to symbolically construct a sparse matrix by repeating
        a vector (again, as far as I can tell).

        :type inputs: scipy.sparse matrix object, shape (n, d)
        :param inputs: sparse matrix on which to compute PCA

        TODO: docstring upgrade. Make it consistent with the numpy/pylearn
        standard.
        """

        # Update component cutoff, in case min_variance or num_components has
        # changed (or both).
        self._update_cutoff()

        Y = structured_dot(inputs, self.W[:, :self.component_cutoff])
        if self.whiten:
            Y /= tensor.sqrt(self.v[:self.component_cutoff])
        return Y

    def function(self, name=None):
        """ Returns a compiled theano function to compute a representation """
        inputs = SparseType('csr', dtype=theano.config.floatX)()
        return theano.function([inputs], self(inputs), name=name)



class PcaOnlineEstimator(object):
    """Online estimation of the leading eigen values/vectors of the covariance
    of some samples.

    Maintains a moving (with discount) low rank (n_eigen) estimate of the
    covariance matrix of some observations. New observations are accumulated
    until the batch is complete, at which point the low rank estimate is
    reevaluated.

    Example:

      pca_esti = pca_online_estimator.PcaOnlineEstimator(dimension_of_the_samples)

      for i in range(number_of_samples):
        pca_esti.observe(samples[i])

      [eigvals, eigvecs] = pca_esti.getLeadingEigen()

    """


    def __init__(self, n_dim, n_eigen = 10, minibatch_size = 25, gamma = 0.999, regularizer = 1e-6, centering = True):
        # dimension of the observations
        self.n_dim = n_dim
        # rank of the low-rank estimate
        self.n_eigen = n_eigen
        # how many observations between reevaluations of the low rank estimate
        self.minibatch_size = minibatch_size
        # the discount factor in the moving estimate
        self.gamma = gamma
        # regularizer of the covariance estimate
        self.regularizer = regularizer
        # wether we center the observations or not: obtain leading eigen of
        # covariance (centering = True) vs second moment (centering = False)
        self.centering = centering

        # Total number of observations: to compute the normalizer for the mean and
        # the covariance.
        self.n_observations = 0
        # Index in the current minibatch
        self.minibatch_index = 0

        # Matrix containing on its *rows*:
        # - the current unnormalized eigen vector estimates
        # - the observations since the last reevaluation
        self.Xt = numpy.zeros([self.n_eigen + self.minibatch_size, self.n_dim])

        # The discounted sum of the observations.
        self.x_sum = numpy.zeros([self.n_dim])

        # The Gram matrix of the observations, ie Xt Xt' (since Xt is rowwise)
        self.G = numpy.zeros([self.n_eigen + self.minibatch_size, self.n_eigen + self.minibatch_size])
        for i in range(self.n_eigen):
            self.G[i,i] = self.regularizer

        # I don't think it's worth "allocating" these 3 next (though they need to be
        # declared). I don't know how to do in place operations...

        # Hold the results of the eigendecomposition of the Gram matrix G
        # (eigen vectors on columns of V).
        self.d = numpy.zeros([self.n_eigen + self.minibatch_size])
        self.V = numpy.zeros([self.n_eigen + self.minibatch_size, self.n_eigen + self.minibatch_size])

        # Holds the unnormalized eigenvectors of the covariance matrix before
        # they're copied back to Xt.
        self.Ut = numpy.zeros([self.n_eigen, self.n_dim])


    def observe(self, x):
        assert(numpy.size(x) == self.n_dim)

        self.n_observations += 1

        # Add the *non-centered* observation to Xt.
        row = self.n_eigen + self.minibatch_index
        self.Xt[row] = x

        # Update the discounted sum of the observations.
        self.x_sum *= self.gamma
        self.x_sum += x

        # To get the mean, we must normalize the sum by:
        # /gamma^(n_observations-1) + /gamma^(n_observations-2) + ... + 1
        normalizer = (1.0 - pow(self.gamma, self.n_observations)) /(1.0 - self.gamma);
        #print "normalizer: ", normalizer

        # Now center the observation.
        # We will lose the first observation as it is the only one in the mean.
        if self.centering:
            self.Xt[row] -= self.x_sum / normalizer

        # Multiply the observation by the discount compensator. Basically
        # we make this observation look "younger" than the previous ones. The actual
        # discount is applied in the reevaluation (and when solving the equations in
        # the case of TONGA) by multiplying every direction with the same aging factor.
        rn = pow(self.gamma, -0.5*(self.minibatch_index+1));
        self.Xt[row] *= rn

        # Update the Gram Matrix.
        # The column.
        self.G[:row+1,row] = numpy.dot( self.Xt[:row+1,:], self.Xt[row,:].transpose() )
        # The symetric row.
        # There are row+1 values, but the diag doesn't need to get copied.
        self.G[row,:row] = self.G[:row,row].transpose()

        self.minibatch_index += 1

        if self.minibatch_index == self.minibatch_size:
            self.reevaluate()


    def reevaluate(self):
        # TODO do the modifications to handle when this is not true.
        assert(self.minibatch_index == self.minibatch_size);

        # Regularize - not necessary but in case
        for i in range(self.n_eigen + self.minibatch_size):
            self.G[i,i] += self.regularizer

        # The Gram matrix is up to date. Get its low rank eigendecomposition.
        # NOTE: the eigenvalues are in ASCENDING order and the vectors are on
        # the columns.
        # With scipy 0.7, you can ask for only some eigenvalues (the n_eigen top
        # ones) but it doesn't look loke it for scipy 0.6.
        self.d, self.V = linalg.eigh(self.G) #, overwrite_a=True)

        # Convert the n_eigen LAST eigenvectors of the Gram matrix contained in V
        # into *unnormalized* eigenvectors U of the covariance (unnormalized wrt
        # the eigen values, not the moving average).
        self.Ut = numpy.dot(self.V[:,-self.n_eigen:].transpose(), self.Xt)

        # Take into account the discount factor.
        # Here, minibatch index is minibatch_size. We age everyone. Because of the
        # previous multiplications to make some observations "younger" we multiply
        # everyone by the same factor.
        # TODO VERIFY THIS!
        rn = pow(self.gamma, -0.5*(self.minibatch_index+1))
        inv_rn2 = 1.0/(rn*rn)
        self.Ut *= 1.0/rn
        self.d *= inv_rn2;

        #print "*** Reevaluate! ***"
        #normalizer = (1.0 - pow(self.gamma, self.n_observations)) /(1.0 - self.gamma)
        #print "normalizer: ", normalizer
        #print self.d / normalizer
        #print self.Ut # unnormalized eigen vectors (wrt eigenvalues AND moving average).

        # Update Xt, G and minibatch_index
        self.Xt[:self.n_eigen,:] = self.Ut

        for i in range(self.n_eigen):
            self.G[i,i] = self.d[-self.n_eigen+i]

        self.minibatch_index = 0

    # Returns a copy of the current estimate of the eigen values and vectors
    # (normalized vectors on rows), normalized by the discounted number of observations.
    def getLeadingEigen(self):
        # We subtract self.minibatch_index in case this call is not right after a reevaluate call.
        normalizer = (1.0 - pow(self.gamma, self.n_observations - self.minibatch_index)) /(1.0 - self.gamma)

        eigvals = self.d[-self.n_eigen:] / normalizer
        eigvecs = numpy.zeros([self.n_eigen, self.n_dim])
        for i in range(self.n_eigen):
            eigvecs[i] = self.Ut[-self.n_eigen+i] / numpy.sqrt(numpy.dot(self.Ut[-self.n_eigen+i], self.Ut[-self.n_eigen+i]))

        return [eigvals, eigvecs]



##################################################
if __name__ == "__main__":
    """
    Load a dataset; compute a PCA transformation matrix from the training
    subset and pickle it (or load a previously computed one); apply said
    transformation to the test and valid subsets.
    """

    import argparse
    from pylearn2.utils import load_data, get_constant

    parser = argparse.ArgumentParser(
        description="Transform the output of a model by Principal Component"
                    " Analysis"
    )
    parser.add_argument('dataset', action='store',
                        type=str,
                        choices=['avicenna', 'harry', 'rita', 'sylvester',
                                 'ule'],
                        help='Dataset on which to compute and apply the PCA')
    parser.add_argument('-i', '--load-file', action='store',
                        type=str,
                        default=None,
                        required=False,
                        help='File containing precomputed PCA (if any)')
    parser.add_argument('-o', '--save-file', action='store',
                        type=str,
                        default='model-pca.pkl',
                        required=False,
                        help='File where the PCA pickle will be saved')
    parser.add_argument('-a', '--algorithm', action='store',
                        type=str,
                        choices=['cov_eig', 'svd', 'online'],
                        default='cov_eig',
                        required=False,
                        help='Which algorithm to use to compute the PCA')
    parser.add_argument('-m', '--minibatch-size', action='store',
                        type=int,
                        default=500,
                        required=False,
                        help='Size of minibatches used in online algorithm')
    parser.add_argument('-n', '--num-components', action='store',
                        type=int,
                        default=None,
                        required=False,
                        help='This many most components will be preserved')
    parser.add_argument('-v', '--min-variance', action='store',
                        type=float,
                        default=0.0,
                        required=False,
                        help="Components with variance below this threshold"
                            " will be discarded")
    parser.add_argument('-w', '--whiten', action='store_const',
                        default=False,
                        const=True,
                        required=False,
                        help='Divide projected features by their '
                             'standard deviation')
    args = parser.parse_args()
    # Load dataset.
    data = load_data({'dataset': args.dataset})
    # TODO: this can be done more efficiently and readably by list
    # comprehensions
    train_data, valid_data, test_data = map(lambda(x):
                                            x.get_value(borrow=True), data)
    print >> sys.stderr, "Dataset shapes:", map(lambda(x):
                                                get_constant(x.shape), data)
    # PCA base-class constructor arguments.
    conf = {
        'num_components': args.num_components,
        'min_variance': args.min_variance,
        'whiten': args.whiten
    }

    # Set PCA subclass from argument.
    if args.algorithm == 'cov_eig':
        PCAImpl = CovEigPCA
    elif args.algorithm == 'svd':
        PCAImpl = SVDPCA
    elif args.algorithm == 'online':
        PCAImpl = OnlinePCA
        conf['minibatch_size'] = args.minibatch_size
    else:
        # This should never happen.
        raise NotImplementedError(args.algorithm)

    # Load precomputed PCA transformation if requested; otherwise compute it.
    if args.load_file:
        pca = Block.load(args.load_file)
    else:
        print "... computing PCA"
        pca = PCAImpl(**conf)
        pca.train(train_data)
        # Save the computed transformation.
        pca.save(args.save_file)

    # Apply the transformation to test and valid subsets.
    inputs = tensor.matrix()
    pca_transform = theano.function([inputs], pca(inputs))
    valid_pca = pca_transform(valid_data)
    test_pca = pca_transform(test_data)
    print >> sys.stderr, "New shapes:", map(numpy.shape, [valid_pca, test_pca])

    # TODO: Compute ALC here when the code using the labels is ready.
