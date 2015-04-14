"""
Training costs for unsupervised learning of energy-based models
"""
from functools import wraps
import logging
import numpy as np
import sys

from theano import scan
import theano.tensor as T
from theano.compat.six.moves import zip as izip

from pylearn2.compat import OrderedDict
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import py_integer_types
from pylearn2.utils.rng import make_theano_rng
from pylearn2.models.rbm import BlockGibbsSampler


logger = logging.getLogger(__name__)

logger.debug("Cost changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when unrolling inference don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)


class NCE(DefaultDataSpecsMixin, Cost):

    """
    Noise-Contrastive Estimation

    See "Noise-Contrastive Estimation: A new estimation principle for
    unnormalized models" by Gutmann and Hyvarinen

    Parameters
    ----------
    noise : WRITEME
        A Distribution from which noisy examples are generated
    noise_per_clean : int
        Number of noisy examples to generate for each clean example given

    """
    def h(self, X, model):
        """
        Computes `h` from the NCE paper.

        Parameters
        ----------
        X : Theano matrix
            Batch of input data
        model : Model
            Any model with a `log_prob` method.

        Returns
        -------
        h : A theano symbol for the `h` function from the paper.
        """
        return - T.nnet.sigmoid(self.G(X, model))

    def G(self, X, model):
        """
        Computes `G` from the NCE paper.

        Parameters
        ----------
        X : Theano matrix
            Batch of input data
        model : Model
            Any model with a `log_prob` method.

        Returns
        -------
        G : A theano symbol for the `G` function from the paper.
        """
        return model.log_prob(X) - self.noise.log_prob(X)

    def expr(self, model, data, noisy_data=None):
        """
        Computes the NCE objective.

        Parameters
        ----------
        model : Model
            Any Model that implements a `log_probs` method.
        data : Theano matrix
        noisy_data : Theano matrix, optional
            The noise samples used for noise-contrastive
            estimation. Will be generated internally if not
            provided. The keyword argument allows FixedVarDescr
            to provide the same noise across several steps of
            a line search.
        """
        space, source = self.get_data_specs(model)
        space.validate(data)
        X = data
        if X.name is None:
            X_name = 'X'
        else:
            X_name = X.name

        m_data = X.shape[0]
        m_noise = m_data * self.noise_per_clean

        if noisy_data is not None:
            space.validate(noisy_data)
            Y = noisy_data
        else:
            Y = self.noise.random_design_matrix(m_noise)

        log_hx = -T.nnet.softplus(-self.G(X, model))
        log_one_minus_hy = -T.nnet.softplus(self.G(Y, model))

        # based on equation 3 of the paper
        # ours is the negative of theirs because
        # they maximize it and we minimize it
        rval = -T.mean(log_hx)-T.mean(log_one_minus_hy)
        rval.name = 'NCE('+X_name+')'

        return rval

    def __init__(self, noise, noise_per_clean):
        self.noise = noise

        assert isinstance(noise_per_clean, py_integer_types)
        self.noise_per_clean = noise_per_clean


class SM(DefaultDataSpecsMixin, Cost):
    """
    (Regularized) Score Matching

    See:

    - "Regularized estimation of image statistics by Score Matching",
      D. Kingma, Y. LeCun, NIPS 2010
    - eqn. 4 of "On Autoencoders and Score Matching for Energy Based Models"
      Swersky et al 2011

    Uses the mean over visible units rather than sum over visible units
    so that hyperparameters won't depend as much on the # of visible units

    Parameters
    ----------
    lambd : WRITEME
    """
    def __init__(self, lambd=0):
        assert lambd >= 0
        self.lambd = lambd

    @wraps(Cost.expr)
    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        X = data
        X_name = 'X' if X.name is None else X.name

        def f(i, _X, _dx):
            return T.grad(_dx[:, i].sum(), _X)[:, i]

        dx = model.score(X)
        ddx, _ = scan(f, sequences=[T.arange(X.shape[1])],
                      non_sequences=[X, dx])
        ddx = ddx.T

        assert len(ddx.type.broadcastable) == 2

        rval = T.mean(0.5 * dx**2 + ddx + self.lambd * ddx**2)
        rval.name = 'sm('+X_name+')'

        return rval


class SMD(DefaultDataSpecsMixin, Cost):
    """
    Denoising Score Matching
    See eqn. 4.3 of
    "A Connection Between Score Matching and Denoising Autoencoders"
    by Pascal Vincent for details

    Note that instead of using half the squared norm we use the mean
    squared error, so that hyperparameters don't depend as much on
    the # of visible units

    Parameters
    ----------
    corruptor : WRITEME
        WRITEME
    """

    def __init__(self, corruptor):
        super(SMD, self).__init__()
        self.corruptor = corruptor

    @wraps(Cost.expr)
    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        X = data
        X_name = 'X' if X.name is None else X.name

        corrupted_X = self.corruptor(X)

        if corrupted_X.name is None:
            corrupted_X.name = 'corrupt('+X_name+')'

        model_score = model.score(corrupted_X)
        assert len(model_score.type.broadcastable) == len(X.type.broadcastable)
        parzen_score = T.grad(
            - T.sum(self.corruptor.corruption_free_energy(corrupted_X, X)),
            corrupted_X)
        assert \
            len(parzen_score.type.broadcastable) == len(X.type.broadcastable)

        score_diff = model_score - parzen_score
        score_diff.name = 'smd_score_diff('+X_name+')'

        assert len(score_diff.type.broadcastable) == len(X.type.broadcastable)

        # TODO: this could probably be faster as a tensordot,
        # but we don't have tensordot for gpu yet
        sq_score_diff = T.sqr(score_diff)

        # sq_score_diff = Print('sq_score_diff',attrs=['mean'])(sq_score_diff)

        smd = T.mean(sq_score_diff)
        smd.name = 'SMD('+X_name+')'

        return smd


class SML(Cost):
    """
    Stochastic Maximum Likelihood

    See "On the convergence of Markovian stochastic algorithms with rapidly
    decreasing ergodicity rates" by Laurent Younes (1998)

    Also known as Persistent Constrastive Divergence (PCD)
    See "Training restricted boltzmann machines using approximations to
    the likelihood gradient" by Tijmen Tieleman  (2008)

    The number of particles fits the batch size.

    Parameters
    ----------
    batch_size: int
        Batch size of the training algorithm
    nsteps: int
        Number of steps made by the block Gibbs sampler between each epoch
    """
    def __init__(self, batch_size, nsteps):
        super(SML, self).__init__()
        self.nchains = batch_size
        self.nsteps = nsteps

    @wraps(Cost.get_gradients)
    def get_gradients(self, model, data, **kwargs):
        cost = self._cost(model, data, **kwargs)

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore',
                       consider_constant=[self.sampler.particles])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        sampler_updates = self.sampler.updates()
        updates.update(sampler_updates)
        return gradients, updates

    def _cost(self, model, data):
        """
        A fake cost that we differentiate symbolically to derive the SML
        update rule.

        Parameters
        ----------
        model : Model
        data : Batch in get_data_specs format

        Returns
        -------
        cost : 0-d Theano tensor
            The fake cost
        """

        if not hasattr(self, 'sampler'):
            self.sampler = BlockGibbsSampler(
                rbm=model,
                particles=0.5+np.zeros((self.nchains, model.get_input_dim())),
                rng=model.rng,
                steps=self.nsteps)

        # compute negative phase updates
        sampler_updates = self.sampler.updates()

        # Compute SML cost
        pos_v = data
        neg_v = self.sampler.particles

        ml_cost = (model.free_energy(pos_v).mean() -
                   model.free_energy(neg_v).mean())

        return ml_cost

    @wraps(Cost.expr)
    def expr(self, model, data):
        return None

    @wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


class CDk(Cost):
    """
    Contrastive Divergence.

    See "Training products of experts by minimizing contrastive divergence"
    by Geoffrey E. Hinton (2002)

    Parameters
    ----------
    nsteps : int
        Number of Markov chain steps for the negative sample
    seed : int
        Seed for the random number generator
    """
    def __init__(self, nsteps, seed=42):
        super(CDk, self).__init__()
        self.nsteps = nsteps
        self.rng = make_theano_rng(seed, which_method='binomial')

    def _cost(self, model, data):
        pos_v = data
        neg_v = data

        for k in range(self.nsteps):
            [neg_v, _locals] = model.gibbs_step_for_v(neg_v, self.rng)

        # Compute CD cost
        ml_cost = (model.free_energy(pos_v).mean() -
                   model.free_energy(neg_v).mean())

        return ml_cost, neg_v

    @wraps(Cost.get_gradients)
    def get_gradients(self, model, data, **kwargs):
        cost, neg_v = self._cost(model, data, **kwargs)

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore',
                       consider_constant=[neg_v])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates

    @wraps(Cost.expr)
    def expr(self, model, data):
        return None

    @wraps(Cost.expr)
    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())
