"""
Costs related to autoencoders.
"""

from functools import wraps

from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor
import theano.sparse

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class GSNFriendlyCost(DefaultDataSpecsMixin, Cost):
    """
    A Cost that can be used to train the GSN.
    """

    @staticmethod
    def cost(target, output):
        """
        The cost of returning `output` when the truth was `target`

        Parameters
        ----------
        target : Theano tensor
            The ground truth
        output : Theano tensor
            The model's output
        """
        raise NotImplementedError()

    def expr(self, model, data, *args, **kwargs):
        """
        The cost of reconstructing `data` using `model`.

        Parameters
        ----------
        model : a GSN
        data : a batch of inputs to reconstruct.
        args : evidently ignored?
        kwargs : optional keyword arguments
            For use with third party TrainingAlgorithms or FixedVarDescr
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        return self.cost(X, model.reconstruct(X))


class MeanSquaredReconstructionError(GSNFriendlyCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    @wraps(GSNFriendlyCost.cost)
    def cost(targets, outputs):
        a = targets
        b = outputs
        return ((a - b) ** 2).sum(axis=1).mean()


class MeanBinaryCrossEntropy(GSNFriendlyCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    @wraps(GSNFriendlyCost.cost)
    def cost(target, output):
        return tensor.nnet.binary_crossentropy(output, target) \
            .sum(axis=1).mean()


class SampledMeanBinaryCrossEntropy(DefaultDataSpecsMixin, Cost):
    """
    .. todo::

        WRITEME properly

    CE cost that goes with sparse autoencoder with L1 regularization on
    activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling

    Parameters
    ----------
    L1 : WRITEME
    ratio : WRITEME
    """

    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.one_ratio = ratio

    @wraps(Cost.expr)
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense = theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1,
                                            prob=self.one_ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of
        # the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P > 0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        reg_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        # params = model.get_params()
        # W = params[2]

        # there is a numerical problem when using
        # tensor.log(1 - model.reconstruct(X, P))
        # Pascal fixed it.
        before_activation = model.reconstruct_without_dec_acti(X, P)

        cost = (1 * X_dense *
                tensor.log(tensor.log(1 + tensor.exp(-1 * before_activation)))
                + (1 - X_dense) *
                tensor.log(1 + tensor.log(1 + tensor.exp(before_activation))))

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * reg_units

        return cost


class SampledMeanSquaredReconstructionError(MeanSquaredReconstructionError):
    """
    mse cost that goes with sparse autoencoder with L1 regularization on
    activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling

    Parameters
    ----------
    L1 : WRITEME
    ratio : WRITEME
    """

    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.ratio = ratio

    @wraps(Cost.expr)
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense = theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1,
                                            prob=self.ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of
        # the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P > 0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        L1_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        # params = model.get_params()
        # W = params[2]
        # L1_weights = theano.tensor.abs_(W).sum()

        cost = ((model.reconstruct(X, P) - X_dense) ** 2)

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * L1_units

        return cost


class SparseActivation(DefaultDataSpecsMixin, Cost):
    """
    Autoencoder sparse activation cost.

    Regularize on KL divergence from desired average activation of each
    hidden unit as described in Andrew Ng's CS294A Lecture Notes. See
    http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf.

    Parameters
    ----------
    coeff : float
        Coefficient for this regularization term in the objective
        function.
    p : float
        Desired average activation of each hidden unit.
    """
    def __init__(self, coeff, p):
        self.coeff = coeff
        self.p = p

    @wraps(Cost.expr)
    def expr(self, model, data, **kwargs):
        X = data
        p = self.p
        p_hat = tensor.abs_(model.encode(X)).mean(axis=0)
        kl = p * tensor.log(p / p_hat) + (1 - p) * \
            tensor.log((1 - p) / (1 - p_hat))
        penalty = self.coeff * kl.sum()
        penalty.name = 'sparse_activation_penalty'
        return penalty
