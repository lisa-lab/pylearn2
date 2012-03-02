"""KMeans as a postprocessing Block subclass."""

import numpy
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
import warnings

try:
    import milk
except:
    milk = None
    warnings.warn(""" Install milk ( http://packages.python.org/milk/ )
                    It has a better k-means implementation. Falling back to
                    our own really slow implementation. """)

class KMeans(Block, Model):
    """
    Block that outputs a vector of probabilities that a sample belong to means
    computed during training.
    """
    def __init__(self, k, nvis, convergence_th=1e-6, max_iter=None, verbose=False):
        """
        Parameters in conf:

        :type k: int
        :param k: number of clusters.

        :type convergence_th: float
        :param convergence_th: threshold of distance to clusters under which
        kmeans stops iterating.

        :type max_iter: int
        :param max_iter: maximum number of iterations. Defaults to infinity.
        """

        Block.__init__(self)
        Model.__init__(self)

        self.input_space = VectorSpace(nvis)

        self.k = k
        self.convergence_th = convergence_th
        if max_iter:
            if max_iter < 0:
                raise Exception('KMeans init: max_iter should be positive.')
            self.max_iter = max_iter
        else:
            self.max_iter = float('inf')

        self.verbose = verbose

    def train(self, dataset, mu=None):
        """
        Process kmeans algorithm on the input to localize clusters.
        """

        #TODO-- why does this sometimes return X and sometimes return nothing?

        X = dataset.get_design_matrix()

        n, m = X.shape
        k = self.k

        if milk is not None:
            #use the milk implementation of k-means if it's available
            cluster_ids, mu = milk.kmeans(X,k)
        else:
            #our own implementation

            # taking random inputs as initial clusters if user does not provide
            # them.
            if mu is not None:
                if not len(mu) == k:
                    raise Exception('You gave %i clusters, but k=%i were expected'
                                    % (len(mu), k))
            else:
                indices = numpy.random.randint(X.shape[0], size=k)
                mu = X[indices]

            try:
                dists = numpy.zeros((n, k))
            except MemoryError:
                print ("dying trying to allocate dists matrix ",
                       "for %d examples and %d means" % (n, k))
                raise

            old_kills = {}

            iter = 0
            mmd = prev_mmd = float('inf')
            while True:
                if self.verbose:
                    print 'kmeans iter ' + str(iter)

                #print 'iter:',iter,' conv crit:',abs(mmd-prev_mmd)
                #if numpy.sum(numpy.isnan(mu)) > 0:
                if numpy.any(numpy.isnan(mu)):
                    print 'nan found'
                    return X

                #computing distances
                for i in xrange(k):
                    dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)

                if iter > 0:
                    prev_mmd = mmd

                min_dists = dists.min(axis=1)

                #mean minimum distance:
                mmd = min_dists.mean()

                print 'cost: ',mmd

                if iter > 0 and (iter >= self.max_iter or \
                                        abs(mmd - prev_mmd) < self.convergence_th):
                    #converged
                    break

                #finding minimum distances
                min_dist_inds = dists.argmin(axis=1)

                #computing means
                i = 0
                blacklist = []
                new_kills = {}
                while i < k:
                    b = min_dist_inds == i
                    if not numpy.any(b):
                        killed_on_prev_iter = True
                        #initializes empty cluster to be the mean of the d data
                        #points farthest from their corresponding means
                        if i in old_kills:
                            d = old_kills[i] - 1
                            if d == 0:
                                d = 50
                            new_kills[i] = d
                        else:
                            d = 5
                        mu[i, :] = 0
                        for j in xrange(d):
                            idx = numpy.argmax(min_dists)
                            min_dists[idx] = 0
                            #chose point idx
                            mu[i, :] += X[idx, :]
                            blacklist.append(idx)
                        mu[i, :] /= float(d)
                        #cluster i was empty, reset it to d far out data points
                        #recomputing distances for this cluster
                        dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)
                        min_dists = dists.min(axis=1)
                        for idx in blacklist:
                            min_dists[idx] = 0
                        min_dist_inds = dists.argmin(axis=1)
                        #done
                        i += 1
                    else:
                        mu[i, :] = numpy.mean(X[b, :], axis=0)
                        if numpy.any(numpy.isnan(mu)):
                            print 'nan found at', i
                            return X
                        i += 1

                old_kills = new_kills

                iter += 1

                """

        self.mu = mu

    def __call__(self, X):
        """
        Compute for each sample its probability to belong to a cluster.

        :type inputs: numpy.ndarray, shape (n, d)
        :param inputs: matrix of samples
        """
        n, m = X.shape
        k = self.k
        mu = self.mu
        dists = numpy.zeros((n, k))
        for i in xrange(k):
            dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)
        return dists / dists.sum(axis=1).reshape(-1, 1)

    def get_weights(self):
        return self.mu

    def get_weights_format(self):
        return ['h','v']

