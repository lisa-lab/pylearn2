# Third-party imports
import numpy
import theano
from theano import tensor
from pylearn.algorithms import pca_online_estimator
from scipy import linalg

# Local imports
from framework.base import Block
from framework.utils import sharedX

floatX = theano.config.floatX

class PCA(Block):
    """
    Block which transforms its input via Principal Component Analysis.
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

        self.num_components = num_components
        self.min_variance = min_variance
        self.whiten = whiten

        self.W = None
        self.v = None
        self.mean = None

        self.component_cutoff = theano.shared(theano._asarray(0, dtype='int64'),
            name='component_cutoff')

        # This module really has no adjustable parameters -- once train()
        # is called once, they are frozen, and are not modified via gradient
        # descent.
        self._params = []

    def train(self, X):
        """
        Compute the PCA transformation matrix.

        Given a rectangular matrix X = USV such that S is a diagonal matrix with
        X's singular values along its diagonal, computes and returns W = V^-1.
        """

        if self.num_components is None:
            self.num_components = X.shape[1]

        # Center each feature.
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
        component_cutoff = self.component_cutoff.get_value()
        self.v.set_value(v[:component_cutoff])
        self.W.set_value(W[:, :component_cutoff])

    def __call__(self, inputs):
        """
        Compute and return the PCA transformation of the current data.

        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix on which to compute PCA
        """

        # Update component cutoff, in case min_variance or num_components has
        # changed (or both).
        self._update_cutoff()

        Y = tensor.dot(inputs - self.mean, self.W[:, :self.component_cutoff])
        if self.whiten:
            Y /= tensor.sqrt(self.v[:self.component_cutoff])
        return Y

    def _update_cutoff(self):
        """
        Update component cutoff shared var, based on current parameters.
        """

        assert self.num_components is not None and self.num_components > 0, \
            'Number of components requested must be >= 1'

        v = self.v.get_value()
        var_mask = v / v.sum() > self.min_variance
        assert numpy.any(var_mask), 'No components exceed the given min. variance'
        var_cutoff = 1 + numpy.where(var_mask)[0].max()

        self.component_cutoff.set_value(min(var_cutoff, self.num_components))

    def _cov_eigen(self, X):
        """
        Compute and return eigen{values,vectors} of X's covariance matrix.

        Returns:
            all eigenvalues in decreasing order
            matrix containing corresponding eigenvectors in its columns
        """
        raise NotImplementedError('_cov_eigen')

class OnlinePCA(PCA):
    def __init__(self, minibatch_size=500, **kwargs):
        super(OnlinePCA, self).__init__(**kwargs)
        self.minibatch_size = minibatch_size

    def _cov_eigen(self, X):
        """
        Perform online computation of covariance matrix eigen{values,vectors}.
        """

        num_components = min(self.num_components, X.shape[1])

        pca_estimator = pca_online_estimator.PcaOnlineEstimator(X.shape[1],
            n_eigen=num_components, minibatch_size=500, centering=False
        )
        for i in range(X.shape[0]):
            pca_estimator.observe(X[i,:])
        v, W = pca_estimator.getLeadingEigen()

        # The resulting components are in *ascending* order of eigenvalue,
        # and W contains eigenvectors in its *rows*, so we reverse both and
        # transpose W.
        return v[::-1], W.T[:, ::-1]

class CovEigPCA(PCA):
    def _cov_eigen(self, X):
        """
        Perform direct computation of covariance matrix eigen{values,vectors}.
        """

        v, W = linalg.eigh(numpy.cov(X.T))

        # The resulting components are in *ascending* order of eigenvalue, and
        # W contains eigenvectors in its *columns*, so we simply reverse both.
        return v[::-1], W[:, ::-1]

class SVDPCA(PCA):
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

##################################################
def get(str):
    """ Evaluate str into an autoencoder object, if it exists """
    obj = globals()[str]
    if issubclass(obj, PCA):
        return obj
    else:
        raise NameError(str)

##################################################


if __name__ == "__main__":
    """
    Load a dataset; compute a PCA transformation matrix from the training subset
    and pickle it (or load a previously computed one); apply said transformation
    to the test and valid subsets.
    """

    from sys import stderr
    import argparse
    from framework.utils import load_data, get_constant

    parser = argparse.ArgumentParser(
        description="Transform the output of a model by Principal Component Analysis"
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
                        help='Size of minibatches used in the online algorithm')
    parser.add_argument('-n', '--num-components', action='store',
                        type=int,
                        default=None,
                        required=False,
                        help='This many most important components will be preserved')
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
                        help='Divide projected features by their standard deviation')
    args = parser.parse_args()

    # Load dataset.
    data = load_data({'dataset': args.dataset})
    [train_data, valid_data, test_data] = map (lambda(x): x.get_value(), data)
    print >> stderr, "Dataset shapes:", map(lambda(x): get_constant(x.shape), data)

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
        pca = PCA.load(args.load_file)
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
    print >> stderr, "New shapes:", map(numpy.shape, [valid_pca, test_pca])

    # TODO: Compute ALC here when the code using the labels is ready.
