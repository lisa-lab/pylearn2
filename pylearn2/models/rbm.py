"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
# Standard library imports
import logging

# Third-party imports
import numpy
N = numpy
np = numpy
import theano
from theano import tensor
from theano.compat.six.moves import zip as izip
T = tensor
from theano.tensor import nnet

# Local imports
from pylearn2.costs.cost import Cost
from pylearn2.blocks import Block, StackedBlocks
from pylearn2.utils import as_floatX, safe_update, sharedX
from pylearn2.models import Model
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_union
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng, make_theano_rng
theano.config.warn.sum_div_dimshuffle_bug = False

logger = logging.getLogger(__name__)

if 0:
    logger.warning('using SLOW rng')
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


def training_updates(visible_batch, model, sampler, optimizer):
    """
    Combine together updates from various sources for RBM training.

    Parameters
    ----------
    visible_batch : tensor_like
        Theano symbolic representing a minibatch on the visible units,
        with the first dimension indexing training examples and the second
        indexing data dimensions.
    model : object
        An instance of `RBM` or a derived class, or one implementing
        the RBM interface.
    sampler : object
        An instance of `Sampler` or a derived class, or one implementing
        the sampler interface.
    optimizer : object
        An instance of `_Optimizer` or a derived class, or one implementing
        the optimizer interface (typically an `_SGDOptimizer`).

    Returns
    -------
    WRITEME
    """
    # TODO: the Optimizer object got deprecated, and this is the only
    #         functionality that requires it. We moved the Optimizer
    #         here with an _ before its name.
    #         We should figure out how best to refactor the code.
    #         Optimizer was problematic because people kept using SGDOptimizer
    #         instead of training_algorithms.sgd.
    # Compute negative phase updates.
    sampler_updates = sampler.updates()
    # Compute SML gradients.
    pos_v = visible_batch
    #neg_v = sampler_updates[sampler.particles]
    neg_v = sampler.particles
    grads = model.ml_gradients(pos_v, neg_v)
    # Build updates dictionary combining (gradient, sampler) updates.
    ups = optimizer.updates(gradients=grads)
    safe_update(ups, sampler_updates)
    return ups


class Sampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.

    Parameters
    ----------
    rbm : object
        An instance of `RBM` or a derived class, or one implementing
        the `gibbs_step_for_v` interface.
    particles : numpy.ndarray
        An initial state for the set of persistent Narkov chain particles
        that will be updated at every step of learning.
    rng : RandomState object
        NumPy random number generator object used to initialize a
        RandomStreams object used in training.
    """
    def __init__(self, rbm, particles, rng):
        self.__dict__.update(rbm=rbm)

        rng = make_np_rng(rng, which_method="randn")
        seed = int(rng.randint(2 ** 30))
        self.s_rng = make_theano_rng(seed, which_method="binomial")
        self.particles = sharedX(particles, name='particles')

    def updates(self):
        """
        Get the dictionary of updates for the sampler's persistent state
        at each step.

        Returns
        -------
        updates : dict
            Dictionary with shared variable instances as keys and symbolic
            expressions indicating how they should be updated as values.

        Notes
        -----
        In the `Sampler` base class, this is simply a stub.
        """
        raise NotImplementedError()


class BlockGibbsSampler(Sampler):
    """
    Implements a persistent Markov chain based on block gibbs sampling
    for use with Persistent Contrastive
    Divergence, a.k.a. stochastic maximum likelhiood, as described in [1].

    .. [1] T. Tieleman. "Training Restricted Boltzmann Machines using
       approximations to the likelihood gradient". Proceedings of the 25th
       International Conference on Machine Learning, Helsinki, Finland,
       2008. http://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf

    Parameters
    ----------
    rbm : object
        An instance of `RBM` or a derived class, or one implementing
        the `gibbs_step_for_v` interface.
    particles : ndarray
        An initial state for the set of persistent Markov chain particles
        that will be updated at every step of learning.
    rng : RandomState object
        NumPy random number generator object used to initialize a
        RandomStreams object used in training.
    steps : int, optional
        Number of Gibbs steps to run the Markov chain for at each
        iteration.
    particles_clip : None or (min, max) pair, optional
        The values of the returned particles will be clipped between
        min and max.
    """
    def __init__(self, rbm, particles, rng, steps=1, particles_clip=None):
        super(BlockGibbsSampler, self).__init__(rbm, particles, rng)
        self.steps = steps
        self.particles_clip = particles_clip

    def updates(self, particles_clip=None):
        """
        Get the dictionary of updates for the sampler's persistent state
        at each step.

        Parameters
        ----------
        particles_clip : WRITEME

        Returns
        -------
        updates : dict
            Dictionary with shared variable instances as keys and symbolic
            expressions indicating how they should be updated as values.
        """
        steps = self.steps
        particles = self.particles
        # TODO: do this with scan?
        for i in xrange(steps):
            particles, _locals = self.rbm.gibbs_step_for_v(
                particles,
                self.s_rng
            )
            assert particles.type.dtype == self.particles.type.dtype
            if self.particles_clip is not None:
                p_min, p_max = self.particles_clip
                # The clipped values should still have the same type
                dtype = particles.dtype
                p_min = tensor.as_tensor_variable(p_min)
                if p_min.dtype != dtype:
                    p_min = tensor.cast(p_min, dtype)
                p_max = tensor.as_tensor_variable(p_max)
                if p_max.dtype != dtype:
                    p_max = tensor.cast(p_max, dtype)
                particles = tensor.clip(particles, p_min, p_max)
        if not hasattr(self.rbm, 'h_sample'):
            self.rbm.h_sample = sharedX(numpy.zeros((0, 0)), 'h_sample')
        return {
            self.particles: particles,
            # TODO: self.rbm.h_sample is never used, why is that here?
            # Moreover, it does not make sense for things like ssRBM.
            self.rbm.h_sample: _locals['h_mean']
        }


class RBM(Block, Model):
    """
    A base interface for RBMs, implementing the binary-binary case.

    Parameters
    ----------
    nvis : int, optional
        Number of visible units in the model.
        (Specifying this implies that the model acts on a vector,
        i.e. it sets vis_space = pylearn2.space.VectorSpace(nvis) )
    nhid : int, optional
        Number of hidden units in the model.
        (Specifying this implies that the model acts on a vector)
    vis_space : pylearn2.space.Space, optional
        Space object describing what kind of vector space the RBM acts
        on. Don't specify if you used nvis / hid
    hid_space: pylearn2.space.Space, optional
        Space object describing what kind of vector space the RBM's
        hidden units live in. Don't specify if you used nvis / nhid
    transformer : WRITEME
    irange : float, optional
        The size of the initial interval around 0 for weights.
    rng : RandomState object or seed, optional
        NumPy RandomState object to use when initializing parameters
        of the model, or (integer) seed to use to create one.
    init_bias_vis : array_like, optional
        Initial value of the visible biases, broadcasted as necessary.
    init_bias_vis_marginals : pylearn2.datasets.dataset.Dataset or None
        Optional. Dataset used to initialize the visible biases to the
        inverse sigmoid of the data marginals
    init_bias_hid : array_like, optional
        initial value of the hidden biases, broadcasted as necessary.
    base_lr : float, optional
        The base learning rate
    anneal_start : int, optional
        Number of steps after which to start annealing on a 1/t schedule
    nchains : int, optional
        Number of negative chains
    sml_gibbs_steps : int, optional
        Number of gibbs steps to take per update
    random_patches_src : pylearn2.datasets.dataset.Dataset or None
        Optional. Dataset from which to draw random patches in order to
        initialize the weights. Patches will be multiplied by irange.
    monitor_reconstruction : bool, optional
        If True, will request a monitoring channel to monitor
        reconstruction error

    Notes
    -----
    The `RBM` class is redundant now that we have a `DBM` class, since
    an RBM is just a DBM with one hidden layer. Users of pylearn2 should
    use single-layer DBMs when possible. Not all RBM functionality has
    been ported to the DBM framework yet, so this is not always possible.
    (Examples: spike-and-slab RBMs, score matching, denoising score matching)
    pylearn2 developers should not add new features to the RBM class or
    add new RBM subclasses. pylearn2 developers should only add documentation
    and bug fixes to the RBM class and subclasses. pylearn2 developers should
    finish porting all RBM functionality to the DBM framework, then turn
    the RBM class into a thin wrapper around the DBM class that allocates
    a single layer DBM.
    """

    def __init__(self, nvis = None, nhid = None,
            vis_space = None,
            hid_space = None,
            transformer = None,
            irange=0.5, rng=None, init_bias_vis = None,
            init_bias_vis_marginals = None, init_bias_hid=0.0,
            base_lr = 1e-3, anneal_start = None, nchains = 100,
            sml_gibbs_steps = 1,
            random_patches_src = None,
            monitor_reconstruction = False):

        Model.__init__(self)
        Block.__init__(self)

        if init_bias_vis_marginals is not None:
            assert init_bias_vis is None
            X = init_bias_vis_marginals.X
            assert X.min() >= 0.0
            assert X.max() <= 1.0

            marginals = X.mean(axis=0)

            #rescale the marginals a bit to avoid NaNs
            init_bias_vis = inverse_sigmoid_numpy(.01 + .98 * marginals)


        if init_bias_vis is None:
            init_bias_vis = 0.0

        rng = make_np_rng(rng, 1001, which_method="uniform")
        self.rng = rng

        if vis_space is None:
            #if we don't specify things in terms of spaces and a transformer,
            #assume dense matrix multiplication and work off of nvis, nhid
            assert hid_space is None
            assert transformer is None or isinstance(transformer,MatrixMul)
            assert nvis is not None
            assert nhid is not None

            if transformer is None:
                if random_patches_src is None:
                    W = rng.uniform(-irange, irange, (nvis, nhid))
                else:
                    if hasattr(random_patches_src, '__array__'):
                        W = irange * random_patches_src.T
                        assert W.shape == (nvis, nhid)
                    else:
                        W = irange * random_patches_src.get_batch_design(
                                nhid).T

                self.transformer = MatrixMul(  sharedX(
                        W,
                        name='W',
                        borrow=True
                    )
                )
            else:
                self.transformer = transformer

            self.vis_space = VectorSpace(nvis)
            self.hid_space = VectorSpace(nhid)
        else:
            assert hid_space is not None
            assert transformer is not None
            assert nvis is None
            assert nhid is None

            self.vis_space = vis_space
            self.hid_space = hid_space
            self.transformer = transformer


        try:
            b_vis = self.vis_space.get_origin()
            b_vis += init_bias_vis
        except ValueError:
            reraise_as(ValueError("bad shape or value for init_bias_vis"))
        self.bias_vis = sharedX(b_vis, name='bias_vis', borrow=True)

        try:
            b_hid = self.hid_space.get_origin()
            b_hid += init_bias_hid
        except ValueError:
            reraise_as(ValueError('bad shape or value for init_bias_hid'))
        self.bias_hid = sharedX(b_hid, name='bias_hid', borrow=True)

        self.random_patches_src = random_patches_src
        self.register_names_to_del(['random_patches_src'])


        self.__dict__.update(nhid=nhid, nvis=nvis)
        self._params = safe_union(self.transformer.get_params(),
                [self.bias_vis, self.bias_hid])

        self.base_lr = base_lr
        self.anneal_start = anneal_start
        self.nchains = nchains
        self.sml_gibbs_steps = sml_gibbs_steps

    def get_default_cost(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError("The RBM class predates the current "
                "Cost-based training algorithms (SGD and BGD). To train "
                "the RBM with PCD, use DefaultTrainingAlgorithm rather "
                "than SGD or BGD. Some RBM subclassess may also be "
                "trained with SGD or BGD by using the "
                "Cost classes defined in pylearn2.costs.ebm_estimation. "
                "Note that it is also possible to make an RBM by allocating "
                "a DBM with only one hidden layer. The DBM class is newer "
                "and supports training with SGD / BGD. In the long run we "
                "should remove the old RBM class and turn it into a wrapper "
                "around the DBM class that makes a 1-layer DBM.")

    def get_input_dim(self):
        """
        Returns
        -------
        dim : int
            The number of elements in the input, if the input is a vector.
        """
        if not isinstance(self.vis_space, VectorSpace):
            raise TypeError("Can't describe " + str(type(self.vis_space))
                    + " as a dimensionality number.")

        return self.vis_space.dim

    def get_output_dim(self):
        """
        Returns
        -------
        dim : int
            The number of elements in the output, if the output is a vector.
        """
        if not isinstance(self.hid_space, VectorSpace):
            raise TypeError("Can't describe " + str(type(self.hid_space))
                    + " as a dimensionality number.")
        return self.hid_space.dim

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.vis_space

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.hid_space

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [param for param in self._params]

    def get_weights(self, borrow=False):
        """
        .. todo::

            WRITEME
        """

        weights ,= self.transformer.get_params()

        return weights.get_value(borrow=borrow)

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """
        return self.transformer.get_weights_topo()

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ['v', 'h']


    def get_monitoring_channels(self, data):
        """
        .. todo::

            WRITEME
        """
        V = data
        theano_rng = make_theano_rng(None, 42, which_method="binomial")

        H = self.mean_h_given_v(V)

        h = H.mean(axis=0)

        return { 'bias_hid_min' : T.min(self.bias_hid),
                 'bias_hid_mean' : T.mean(self.bias_hid),
                 'bias_hid_max' : T.max(self.bias_hid),
                 'bias_vis_min' : T.min(self.bias_vis),
                 'bias_vis_mean' : T.mean(self.bias_vis),
                 'bias_vis_max': T.max(self.bias_vis),
                 'h_min' : T.min(h),
                 'h_mean': T.mean(h),
                 'h_max' : T.max(h),
                'reconstruction_error' : self.reconstruction_error(V,
                    theano_rng) }

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channel.

        This implementation returns specification corresponding to unlabeled
        inputs.

        Returns
        -------
        WRITEME
        """
        return (self.get_input_space(), self.get_input_source())

    def ml_gradients(self, pos_v, neg_v):
        """
        Get the contrastive gradients given positive and negative phase
        visible units.

        Parameters
        ----------
        pos_v : tensor_like
            Theano symbolic representing a minibatch on the visible units,
            with the first dimension indexing training examples and the
            second indexing data dimensions (usually actual training data).
        neg_v : tensor_like
            Theano symbolic representing a minibatch on the visible units,
            with the first dimension indexing training examples and the
            second indexing data dimensions (usually reconstructions of the
            data or sampler particles from a persistent Markov chain).

        Returns
        -------
        grads : list
            List of Theano symbolic variables representing gradients with
            respect to model parameters, in the same order as returned by
            `params()`.

        Notes
        -----
        `pos_v` and `neg_v` need not have the same first dimension, i.e.
        minibatch size.
        """

        # taking the mean over each term independently allows for different
        # mini-batch sizes in the positive and negative phase.
        ml_cost = (self.free_energy_given_v(pos_v).mean() -
                   self.free_energy_given_v(neg_v).mean())

        grads = tensor.grad(ml_cost, self.get_params(),
                            consider_constant=[pos_v, neg_v])

        return grads


    def train_batch(self, dataset, batch_size):
        """
        .. todo::

            WRITEME properly

        A default learning rule based on SML
        """
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
        return True

    def learn_mini_batch(self, X):
        """
        .. todo::

            WRITEME

        A default learning rule based on SML
        """

        if not hasattr(self, 'learn_func'):
            self.redo_theano()

        rval =  self.learn_func(X)

        return rval

    def redo_theano(self):
        """
        Compiles the theano function for the default learning rule
        """

        init_names = dir(self)

        minibatch = tensor.matrix()

        optimizer = _SGDOptimizer(self, self.base_lr, self.anneal_start)

        sampler = sampler = BlockGibbsSampler(self, 0.5 + np.zeros((
            self.nchains, self.get_input_dim())), self.rng,
            steps= self.sml_gibbs_steps)


        updates = training_updates(visible_batch=minibatch, model=self,
                                   sampler=sampler, optimizer=optimizer)

        self.learn_func = theano.function([minibatch], updates=updates)

        final_names = dir(self)

        self.register_names_to_del([name for name in final_names
            if name not in init_names])

    def gibbs_step_for_v(self, v, rng):
        """
        Do a round of block Gibbs sampling given visible configuration

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch
            of training examples (or negative phase particles), with the
            first dimension indexing training examples and the second
            indexing data dimensions.
        rng : RandomStreams object
            Random number generator to use for sampling the hidden and
            visible units.

        Returns
        -------
        v_sample : tensor_like
            Theano symbolic representing the new visible unit state after one
            round of Gibbs sampling.
        locals : dict
            Contains the following auxiliary state as keys (all symbolics
            except shape tuples):

              * `h_mean`: the returned value from `mean_h_given_v`
              * `h_mean_shape`: shape tuple indicating the size of
                `h_mean` and `h_sample`
              * `h_sample`: the stochastically sampled hidden units
              * `v_mean_shape`: shape tuple indicating the shape of
                `v_mean` and `v_sample`
              * `v_mean`: the returned value from `mean_v_given_h`
              * `v_sample`: the stochastically sampled visible units
        """
        h_mean = self.mean_h_given_v(v)
        assert h_mean.type.dtype == v.type.dtype
        # For binary hidden units
        # TODO: factor further to extend to other kinds of hidden units
        #       (e.g. spike-and-slab)
        h_sample = rng.binomial(size = h_mean.shape, n = 1 , p = h_mean,
            dtype=h_mean.type.dtype)
        assert h_sample.type.dtype == v.type.dtype
        # v_mean is always based on h_sample, not h_mean, because we don't
        # want h transmitting more than one bit of information per unit.
        v_mean = self.mean_v_given_h(h_sample)
        assert v_mean.type.dtype == v.type.dtype
        v_sample = self.sample_visibles([v_mean], v_mean.shape, rng)
        assert v_sample.type.dtype == v.type.dtype
        return v_sample, locals()

    def sample_visibles(self, params, shape, rng):
        """
        Stochastically sample the visible units given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        params : list
            List of the necessary parameters to sample :math:`p(v|h)`. In the
            case of a binary-binary RBM this is a single-element list
            containing the symbolic representing :math:`p(v|h)`, as returned
            by `mean_v_given_h`.

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing stochastic samples from :math:`p(v|h)`
        """
        v_mean = params[0]
        return as_floatX(rng.uniform(size=shape) < v_mean)

    def input_to_h_from_v(self, v):
        """
        Compute the affine function (linear map plus bias) that serves as
        input to the hidden layer in an RBM.

        Parameters
        ----------
        v : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the one or several
            minibatches on the visible units, with the first dimension
            indexing training examples and the second indexing data dimensions.

        Returns
        -------
        a : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input to each
            hidden unit for each training example.
        """

        if isinstance(v, tensor.Variable):
            return self.bias_hid + self.transformer.lmul(v)
        else:
            return [self.input_to_h_from_v(vis) for vis in v]

    def input_to_v_from_h(self, h):
        """
        Compute the affine function (linear map plus bias) that serves as
        input to the visible layer in an RBM.

        Parameters
        ----------
        h : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the one or several
            minibatches on the hidden units, with the first dimension
            indexing training examples and the second indexing data dimensions.

        Returns
        -------
        a : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input to each
            visible unit for each row of h.
        """
        if isinstance(h, tensor.Variable):
            return self.bias_vis + self.transformer.lmul_T(h)
        else:
            return [self.input_to_v_from_h(hid) for hid in h]

    def upward_pass(self, v):
        """
        Wrapper around mean_h_given_v method.  Called when RBM is accessed
        by mlp.HiddenLayer.
        """
        return self.mean_h_given_v(v)

    def mean_h_given_v(self, v):
        """
        Compute the mean activation of the hidden units given visible unit
        configurations for a set of training examples.

        Parameters
        ----------
        v : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the hidden unit
            states for a batch (or several) of training examples, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        h : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the mean
            (deterministic) hidden unit activations given the visible units.
        """
        if isinstance(v, tensor.Variable):
            return nnet.sigmoid(self.input_to_h_from_v(v))
        else:
            return [self.mean_h_given_v(vis) for vis in v]

    def mean_v_given_h(self, h):
        """
        Compute the mean activation of the visibles given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        h : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the hidden unit
            states for a batch (or several) of training examples, with the
            first dimension indexing training examples and the second
            indexing hidden units.

        Returns
        -------
        vprime : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the mean
            (deterministic) reconstruction of the visible units given the
            hidden units.
        """
        if isinstance(h, tensor.Variable):
            return nnet.sigmoid(self.input_to_v_from_h(h))
        else:
            return [self.mean_v_given_h(hid) for hid in h]

    def free_energy_given_v(self, v):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch
            of training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        f : tensor_like
            1-dimensional tensor (vector) representing the free energy
            associated with each row of v.
        """
        sigmoid_arg = self.input_to_h_from_v(v)
        return (-tensor.dot(v, self.bias_vis) -
                 nnet.softplus(sigmoid_arg).sum(axis=1))

    def free_energy(self, V):
        return self.free_energy_given_v(V)


    def free_energy_given_h(self, h):
        """
        Calculate the free energy of a hidden unit configuration by
        marginalizing over the visible units.

        Parameters
        ----------
        h : tensor_like
            Theano symbolic representing the hidden unit states, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        f : tensor_like
            1-dimensional tensor (vector) representing the free energy
            associated with each row of v.
        """
        sigmoid_arg = self.input_to_v_from_h(h)
        return (-tensor.dot(h, self.bias_hid) -
                nnet.softplus(sigmoid_arg).sum(axis=1))

    def __call__(self, v):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.

        This just aliases the `mean_h_given_v()` function for syntactic
        sugar/convenience.
        """
        return self.mean_h_given_v(v)

    def reconstruction_error(self, v, rng):
        """
        Compute the mean-squared error (mean over examples, sum over units)
        across a minibatch after a Gibbs step starting from the training data.

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch
            of training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.
        rng : RandomStreams object
            Random number generator to use for sampling the hidden and
            visible units.

        Returns
        -------
        mse : tensor_like
            0-dimensional tensor (essentially a scalar) indicating the mean
            reconstruction error across the minibatch.

        Notes
        -----
        The reconstruction used to assess error samples only the hidden
        units. For the visible units, it uses the conditional mean. No sampling
        of the visible units is done, to reduce noise in the estimate.
        """
        sample, _locals = self.gibbs_step_for_v(v, rng)
        return ((_locals['v_mean'] - v) ** 2).sum(axis=1).mean()


class GaussianBinaryRBM(RBM):
    """
    An RBM with Gaussian visible units and binary hidden units.

    Parameters
    ----------
    energy_function_class : WRITEME
    nvis : int, optional
        Number of visible units in the model.
    nhid : int, optional
        Number of hidden units in the model.
    vis_space : WRITEME
    hid_space : WRITEME
    irange : float, optional
        The size of the initial interval around 0 for weights.
    rng : RandomState object or seed, optional
        NumPy RandomState object to use when initializing parameters
        of the model, or (integer) seed to use to create one.
    mean_vis : bool, optional
        Don't actually sample visibles; make sample method simply return
        mean.
    init_sigma : float or numpy.ndarray, optional
        Initial value of the sigma variable. If init_sigma is a scalar
        and sigma is not, will be broadcasted.
    learn_sigma : bool, optional
        WRITEME
    sigma_lr_scale : float, optional
        WRITEME
    init_bias_hid : scalar or 1-d array of length `nhid`
        Initial value for the biases on hidden units.
    min_sigma, max_sigma : float, float, optional
        Elements of sigma are clipped to this range during learning
    """

    def __init__(self, energy_function_class,
            nvis = None,
            nhid = None,
            vis_space = None,
            hid_space = None,
            transformer = None,
            irange=0.5, rng=None,
                 mean_vis=False, init_sigma=2., learn_sigma=False,
                 sigma_lr_scale=1., init_bias_hid=0.0,
                 min_sigma = .1, max_sigma = 10.):
        super(GaussianBinaryRBM, self).__init__(nvis = nvis, nhid = nhid,
                                                transformer = transformer,
                                                vis_space = vis_space,
                                                hid_space = hid_space,
                                                irange = irange, rng = rng,
                                                init_bias_hid = init_bias_hid)

        self.learn_sigma = learn_sigma
        self.init_sigma = init_sigma
        self.sigma_lr_scale = float(sigma_lr_scale)

        if energy_function_class.supports_vector_sigma():
            base = N.ones(nvis)
        else:
            base = 1

        self.sigma_driver = sharedX(
            base * init_sigma / self.sigma_lr_scale,
            name='sigma_driver',
            borrow=True
        )

        self.sigma = self.sigma_driver * self.sigma_lr_scale
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        if self.learn_sigma:
            self._params.append(self.sigma_driver)

        self.mean_vis = mean_vis

        self.energy_function = energy_function_class(
                    transformer = self.transformer,
                    sigma=self.sigma,
                    bias_vis=self.bias_vis,
                    bias_hid=self.bias_hid
                )

    def _modify_updates(self, updates):
        """
        .. todo::

            WRITEME
        """
        if self.sigma_driver in updates:
            assert self.learn_sigma
            updates[self.sigma_driver] = T.clip(
                updates[self.sigma_driver],
                self.min_sigma / self.sigma_lr_scale,
                self.max_sigma / self.sigma_lr_scale
            )

    def score(self, V):
        """
        .. todo::

            WRITEME
        """
        return self.energy_function.score(V)

    def P_H_given_V(self, V):
        """
        .. todo::

            WRITEME
        """
        return self.energy_function.mean_H_given_V(V)

    def mean_h_given_v(self, v):
        """
        .. todo::

            WRITEME
        """
        return self.P_H_given_V(v)

    def mean_v_given_h(self, h):
        """
        Compute the mean activation of the visibles given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        h : tensor_like
            Theano symbolic representing the hidden unit states for a batch
            of training examples, with the first dimension indexing training
            examples and the second indexing hidden units.

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing the mean (deterministic)
            reconstruction of the visible units given the hidden units.
        """

        return self.energy_function.mean_V_given_H(h)
        #return self.bias_vis + self.sigma * tensor.dot(h, self.weights.T)

    def free_energy_given_v(self, V):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch
            of training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        f : tensor_like
            1-dimensional tensor representing the free energy of the visible
            unit configuration for each example in the batch
        """

        """hid_inp = self.input_to_h_from_v(v)
        squared_term = ((self.bias_vis - v) ** 2.) / (2. * self.sigma)
        rval =  squared_term.sum(axis=1) - nnet.softplus(hid_inp).sum(axis=1)
        assert len(rval.type.broadcastable) == 1"""

        return self.energy_function.free_energy(V)

    def free_energy(self, V):
        """
        .. todo::

            WRITEME
        """
        return self.energy_function.free_energy(V)

    def sample_visibles(self, params, shape, rng):
        """
        Stochastically sample the visible units given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        params : list
            List of the necessary parameters to sample :math:`p(v|h)`.
            In the case of a Gaussian-binary RBM this is a single-element
            list containing the conditional mean.
        shape : WRITEME
        rng : WRITEME

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing stochastic samples from
            :math:`p(v|h)`

        Notes
        -----
        If `mean_vis` is specified as `True` in the constructor, this is
        equivalent to a call to `mean_v_given_h`.
        """
        v_mean = params[0]
        if self.mean_vis:
            return v_mean
        else:
            # zero mean, std sigma noise
            zero_mean = rng.normal(size=shape) * self.sigma
            return zero_mean + v_mean


class mu_pooled_ssRBM(RBM):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    alpha : WRITEME
        Vector of length nslab, diagonal precision term on s.
    b : WRITEME
        Vector of length nhid, hidden unit bias.
    B : WRITEME
        Vector of length nvis, diagonal precision on v.  Lambda in ICML2011
        paper.
    Lambda : WRITEME
        Matrix of shape nvis x nhid, whose i-th column encodes a diagonal
        precision on v, conditioned on h_i.  phi in ICML2011 paper.
    log_alpha : WRITEME
        Vector of length nslab, precision on s.
    mu : WRITEME
        Vector of length nslab, mean parameter on s.
    W : WRITEME
        Matrix of shape nvis x nslab, weights of the nslab linear filters s.
    """

    def __init__(self, nvis, nhid, n_s_per_h,
            batch_size,
            alpha0, alpha_irange,
            b0,
            B0,
            Lambda0, Lambda_irange,
            mu0,
            W_irange=None,
            rng=None):

        rng = make_np_rng(rng, 1001, which_method="rand")

        self.nhid = nhid
        self.nslab = nhid * n_s_per_h
        self.n_s_per_h = n_s_per_h
        self.nvis = nvis

        self.batch_size = batch_size

        # configure \alpha: precision parameter on s
        alpha_init = numpy.zeros(self.nslab) + alpha0
        if alpha_irange > 0:
            alpha_init += (2 * rng.rand(self.nslab) - 1) * alpha_irange
        self.log_alpha = sharedX(numpy.log(alpha_init), name='log_alpha')
        self.alpha = tensor.exp(self.log_alpha)
        self.alpha.name = 'alpha'

        self.mu = sharedX(
                numpy.zeros(self.nslab) + mu0,
                name='mu', borrow=True)
        self.b = sharedX(
                numpy.zeros(self.nhid) + b0,
                name='b', borrow=True)

        if W_irange is None:
            # Derived closed to Xavier Glorot's magic formula
            W_irange = 2 / numpy.sqrt(nvis * nhid)
        self.W = sharedX(
                (.5 - rng.rand(self.nvis, self.nslab)) * 2 * W_irange,
                name='W', borrow=True)

        # THE BETA IS IGNORED DURING TRAINING - FIXED AT MARGINAL DISTRIBUTION
        self.B = sharedX(numpy.zeros(self.nvis) + B0, name='B', borrow=True)

        if Lambda_irange > 0:
            L = (rng.rand(self.nvis, self.nhid) * Lambda_irange
                    + Lambda0)
        else:
            L = numpy.zeros((self.nvis, self.nhid)) + Lambda0
        self.Lambda = sharedX(L, name='Lambda', borrow=True)

        self._params = [
                self.mu,
                self.B,
                self.Lambda,
                self.W,
                self.b,
                self.log_alpha]

    #def ml_gradients(self, pos_v, neg_v):
    #    inherited version is OK.

    def gibbs_step_for_v(self, v, rng):
        """
        .. todo::

            WRITEME
        """
        # Sometimes, the number of examples in the data set is not a
        # multiple of self.batch_size.
        batch_size = v.shape[0]

        # sample h given v
        h_mean = self.mean_h_given_v(v)
        h_mean_shape = (batch_size, self.nhid)
        h_sample = rng.binomial(size=h_mean_shape,
                n = 1, p = h_mean, dtype = h_mean.dtype)

        # sample s given (v,h)
        s_mu, s_var = self.mean_var_s_given_v_h1(v)
        s_mu_shape = (batch_size, self.nslab)
        s_sample = s_mu + rng.normal(size=s_mu_shape) * tensor.sqrt(s_var)
        #s_sample=(s_sample.reshape()*h_sample.dimshuffle(0,1,'x')).flatten(2)

        # sample v given (s,h)
        v_mean, v_var = self.mean_var_v_given_h_s(h_sample, s_sample)
        v_mean_shape = (batch_size, self.nvis)
        v_sample = rng.normal(size=v_mean_shape) * tensor.sqrt(v_var) + v_mean

        del batch_size
        return v_sample, locals()

    ## TODO?
    def sample_visibles(self, params, shape, rng):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError('mu_pooled_ssRBM.sample_visibles')

    def input_to_h_from_v(self, v):
        """
        .. todo::

            WRITEME
        """
        D = self.Lambda
        alpha = self.alpha

        def sum_s(x):
            return x.reshape((
                -1,
                self.nhid,
                self.n_s_per_h)).sum(axis=2)

        return tensor.add(
                self.b,
                -0.5 * tensor.dot(v * v, D),
                sum_s(self.mu * tensor.dot(v, self.W)),
                sum_s(0.5 * tensor.sqr(tensor.dot(v, self.W)) / alpha))

    #def mean_h_given_v(self, v):
    #    inherited version is OK:
    #    return nnet.sigmoid(self.input_to_h_from_v(v))

    def mean_var_v_given_h_s(self, h, s):
        """
        .. todo::

            WRITEME
        """
        v_var = 1 / (self.B + tensor.dot(h, self.Lambda.T))
        s3 = s.reshape((
                -1,
                self.nhid,
                self.n_s_per_h))
        hs = h.dimshuffle(0, 1, 'x') * s3
        v_mu = tensor.dot(hs.flatten(2), self.W.T) * v_var
        return v_mu, v_var

    def mean_var_s_given_v_h1(self, v):
        """
        .. todo::

            WRITEME
        """
        alpha = self.alpha
        return (self.mu + tensor.dot(v, self.W) / alpha,
                1.0 / alpha)

    ## TODO?
    def mean_v_given_h(self, h):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError('mu_pooled_ssRBM.mean_v_given_h')

    def free_energy_given_v(self, v):
        """
        .. todo::

            WRITEME
        """
        sigmoid_arg = self.input_to_h_from_v(v)
        return tensor.add(
                0.5 * (self.B * (v ** 2)).sum(axis=1),
                -tensor.nnet.softplus(sigmoid_arg).sum(axis=1))

    #def __call__(self, v):
    #    inherited version is OK

    #def reconstruction_error:
    #    inherited version should be OK

    #def params(self):
    #    inherited version is OK.


def build_stacked_RBM(nvis, nhids, batch_size, vis_type='binary',
        input_mean_vis=None, irange=1e-3, rng=None):
    """
    .. todo::

        WRITEME properly

    Note from IG:
        This method doesn't seem to work correctly with Gaussian RBMs.
        In general, this is a difficult function to support, because it
        needs to pass the write arguments to the constructor of many kinds
        of RBMs. It would probably be better to just construct an instance
        of pylearn2.models.mlp.MLP with its hidden layers set to instances
        of pylearn2.models.mlp.RBM_Layer. If anyone is working on this kind
        of problem, a PR replacing this function with a helper function to
        make such an MLP would be very welcome.


    Allocate a StackedBlocks containing RBMs.

    The visible units of the input RBM can be either binary or gaussian,
    the other ones are all binary.
    """
    #TODO: not sure this is the right way of dealing with mean_vis.
    layers = []
    assert vis_type in ['binary', 'gaussian']
    if vis_type == 'binary':
        assert input_mean_vis is None
    elif vis_type == 'gaussian':
        assert input_mean_vis in (True, False)

    # The number of visible units in each layer is the initial input
    # size and the first k-1 hidden unit sizes.
    nviss = [nvis] + nhids[:-1]
    seq = izip(
            xrange(len(nhids)),
            nhids,
            nviss,
            )
    for k, nhid, nvis in seq:
        if k == 0 and vis_type == 'gaussian':
            rbm = GaussianBinaryRBM(nvis=nvis, nhid=nhid,
                    batch_size=batch_size,
                    irange=irange,
                    rng=rng,
                    mean_vis=input_mean_vis)
        else:
            rbm = RBM(nvis - nvis, nhid=nhid,
                    batch_size=batch_size,
                    irange=irange,
                    rng=rng)
        layers.append(rbm)

    # Create the stack
    return StackedBlocks(layers)


class L1_ActivationCost(Cost):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    target : WRITEME
    eps : WRITEME
    coeff : WRITEME
    """
    def __init__(self, target, eps, coeff):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        H = model.P_H_given_V(X)
        h = H.mean(axis=0)
        err = abs(h - self.target)
        dead = T.maximum(err - self.eps, 0.)
        assert dead.ndim == 1
        rval = self.coeff * dead.mean()
        return rval

    def get_data_specs(self, model):
        """
        .. todo::

            WRITEME
        """
        return (model.get_input_space(), model.get_input_source())


# The following functionality was deprecated, but is evidently
# still needed to make the RBM work

class _Optimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """

    def updates(self):
        """Return symbolic updates to apply."""
        raise NotImplementedError()


class _SGDOptimizer(_Optimizer):
    """
    Compute updates by stochastic gradient descent on mini-batches.

    Supports constant learning rates, or decreasing like 1/t after an initial
    period.

    Parameters
    ----------
    params : object or list
        Either a Model object with a .get_params() method, or a list of
        parameters to be optimized.
    base_lr : float
        The base learning rate before annealing or parameter-specific
        scaling.
    anneal_start : int, optional
        Number of steps after which to start annealing the learning
        rate at a 1/t schedule, where t is the number of stochastic
        gradient updates.
    use_adagrad : bool, optional
        'adagrad' adaptive learning rate scheme is used. If set to True,
        base_lr is used as e0.
    kwargs : dict
        WRITEME

    Notes
    -----
    The formula to compute the effective learning rate on a parameter is:
    <paramname>_lr * max(0.0, min(base_lr, lr_anneal_start/(iteration+1)))

    Parameter-specific learning rates can be set by passing keyword
    arguments <name>_lr, where name is the .name attribute of a given
    parameter.

    Parameter-specific bounding values can be specified by passing
    keyword arguments <param>_clip, which should be a (min, max) pair.

    Adagrad is recommended with sparse inputs. It normalizes the base
    learning rate of a parameter theta_i by the accumulated 2-norm of its
    gradient: e{ti} = e0 / sqrt( sum_t (dL_t / dtheta_i)^2 )
    """
    def __init__(self, params, base_lr, anneal_start=None, use_adagrad=False,
                 ** kwargs):
        if hasattr(params, '__iter__'):
            self.params = params
        elif hasattr(params, 'get_params') and hasattr(
                params.get_params, '__call__'):
            self.params = params.get_params()
        else:
            raise ValueError("SGDOptimizer couldn't figure out what to do "
                             "with first argument: '%s'" % str(params))
        if anneal_start == None:
            self.anneal_start = None
        else:
            self.anneal_start = as_floatX(anneal_start)

        # Create accumulators and epsilon0's
        self.use_adagrad = use_adagrad
        if self.use_adagrad:
            self.accumulators = {}
            self.e0s = {}
            for param in self.params:
                self.accumulators[param] = theano.shared(
                        value=as_floatX(0.), name='acc_%s' % param.name)
                self.e0s[param] = as_floatX(base_lr)

        # Set up the clipping values
        self.clipping_values = {}
        # Keep track of names already seen
        clip_names_seen = set()
        for parameter in self.params:
            clip_name = '%s_clip' % parameter.name
            if clip_name in kwargs:
                if clip_name in clip_names_seen:
                    logger.warning('In SGDOptimizer, at least two parameters '
                                   'have the same name. Both will be affected '
                                   'by the keyword argument '
                                   '{0}.'.format(clip_name))
                clip_names_seen.add(clip_name)
                p_min, p_max = kwargs[clip_name]
                assert p_min <= p_max
                self.clipping_values[parameter] = (p_min, p_max)

        # Check that no ..._clip keyword is being ignored
        for clip_name in clip_names_seen:
            kwargs.pop(clip_name)
        for kw in kwargs.iterkeys():
            if kw[-5:] == '_clip':
                logger.warning('In SGDOptimizer, keyword argument {0} '
                               'will be ignored, because no parameter '
                               'was found with name {1}.'.format(kw, kw[:-5]))

        self.learning_rates_setup(base_lr, **kwargs)

    def learning_rates_setup(self, base_lr, **kwargs):
        """
        Initializes parameter-specific learning rate dictionary and shared
        variables for the annealed base learning rate and iteration number.

        Parameters
        ----------
        base_lr : float
            The base learning rate before annealing or parameter-specific
            scaling.
        kwargs : dict
            WRITEME

        Notes
        -----
        Parameter-specific learning rates can be set by passing keyword
        arguments <name>_lr, where name is the .name attribute of a given
        parameter.
        """
        # Take care of learning rate scales for individual parameters
        self.learning_rates = {}
        # Base learning rate per example.
        self.base_lr = theano._asarray(base_lr, dtype=theano.config.floatX)

        # Keep track of names already seen
        lr_names_seen = set()
        for parameter in self.params:
            lr_name = '%s_lr' % parameter.name
            if lr_name in lr_names_seen:
                logger.warning('In SGDOptimizer, '
                               'at least two parameters have the same name. '
                               'Both will be affected by the keyword argument '
                               '{0}.'.format(lr_name))
            lr_names_seen.add(lr_name)

            thislr = kwargs.get(lr_name, 1.)
            self.learning_rates[parameter] = sharedX(thislr, lr_name)

        # Verify that no ..._lr keyword argument is ignored
        for lr_name in lr_names_seen:
            if lr_name in kwargs:
                kwargs.pop(lr_name)
        for kw in kwargs.iterkeys():
            if kw[-3:] == '_lr':
                logger.warning('In SGDOptimizer, keyword argument {0} '
                               'will be ignored, because no parameter '
                               'was found with name {1}.'.format(kw, kw[:-3]))

        # A shared variable for storing the iteration number.
        self.iteration = sharedX(theano._asarray(0, dtype='int32'),
                                 name='iter')

        # A shared variable for storing the annealed base learning rate, used
        # to lower the learning rate gradually after a certain amount of time.
        self.annealed = sharedX(base_lr, 'annealed')

    def learning_rate_updates(self, gradients):
        """
        Compute a dictionary of shared variable updates related to annealing
        the learning rate.

        Parameters
        ----------
        gradients : WRITEME

        Returns
        -------
        updates : dict
            A dictionary with the shared variables representing SGD metadata
            as keys and a symbolic expression of how they are to be updated as
            values.
        """
        ups = {}

        if self.use_adagrad:
            learn_rates = []
            for param, gp in zip(self.params, gradients):
                acc = self.accumulators[param]
                ups[acc] = acc + (gp ** 2).sum()
                learn_rates.append(self.e0s[param] / (ups[acc] ** .5))
        else:
            # Annealing coefficient. Here we're using a formula of
            # min(base_lr, anneal_start / (iteration + 1))
            if self.anneal_start is None:
                annealed = sharedX(self.base_lr)
            else:
                frac = self.anneal_start / (self.iteration + 1.)
                annealed = tensor.minimum(
                                          as_floatX(frac),
                                          self.base_lr  # maximum learning rate
                                          )

            # Update the shared variable for the annealed learning rate.
            ups[self.annealed] = annealed
            ups[self.iteration] = self.iteration + 1

            # Calculate the learning rates for each parameter, in the order
            # they appear in self.params
            learn_rates = [annealed * self.learning_rates[p] for p in
                    self.params]
        return ups, learn_rates

    def updates(self, gradients):
        """
        Return symbolic updates to apply given a set of gradients
        on the parameters being optimized.

        Parameters
        ----------
        gradients : list of tensor_likes
            List of symbolic gradients for the parameters contained
            in self.params, in the same order as in self.params.

        Returns
        -------
        updates : dict
            A dictionary with the shared variables in self.params as keys
            and a symbolic expression of how they are to be updated each
            SGD step as values.

        Notes
        -----
        `cost_updates` is a convenient helper function that takes all
        necessary gradients with respect to a given symbolic cost.
        """
        ups = {}
        # Add the learning rate/iteration updates
        l_ups, learn_rates = self.learning_rate_updates(gradients)
        safe_update(ups, l_ups)

        # Get the updates from sgd_updates, a PyLearn library function.
        p_up = dict(self.sgd_updates(self.params, gradients, learn_rates))

        # Add the things in p_up to ups
        safe_update(ups, p_up)

        # Clip the values if needed.
        # We do not want the clipping values to force an upcast
        # of the update: updates should have the same type as params
        for param, (p_min, p_max) in self.clipping_values.iteritems():
            p_min = tensor.as_tensor(p_min)
            p_max = tensor.as_tensor(p_max)
            dtype = param.dtype
            if p_min.dtype != dtype:
                p_min = tensor.cast(p_min, dtype)
            if p_max.dtype != dtype:
                p_max = tensor.cast(p_max, dtype)
            ups[param] = tensor.clip(ups[param], p_min, p_max)

        # Return the updates dictionary.
        return ups

    def cost_updates(self, cost):
        """
        Return symbolic updates to apply given a cost function.

        Parameters
        ----------
        cost : tensor_like
            Symbolic cost with respect to which the gradients of
            the parameters should be taken. Should be 0-dimensional
            (scalar valued).

        Returns
        -------
        updates : dict
            A dictionary with the shared variables in self.params as keys
            and a symbolic expression of how they are to be updated each
            SGD step as values.
        """
        grads = [tensor.grad(cost, p) for p in self.params]
        return self.updates(gradients=grads)

    def sgd_updates(self, params, grads, stepsizes):
        """
        Return a list of (pairs) that can be used
        as updates in theano.function to
        implement stochastic gradient descent.

        Parameters
        ----------
        params : list of Variable
            variables to adjust in order to minimize some cost
        grads : list of Variable
            the gradient on each param (with respect to some cost)
        stepsizes : symbolic scalar or list of one symbolic scalar per param
            step by this amount times the negative gradient on each iteration
        """
        try:
            iter(stepsizes)
        except Exception:
            stepsizes = [stepsizes for p in params]
        if len(params) != len(grads):
            raise ValueError('params and grads have different lens')
        updates = [(p, p - step * gp) for (step, p, gp)
                in zip(stepsizes, params, grads)]
        return updates

    def sgd_momentum_updates(self, params, grads, stepsizes, momentum=0.9):
        """
        .. todo::

            WRITEME
        """
        # if stepsizes is just a scalar, expand it to match params
        try:
            iter(stepsizes)
        except Exception:
            stepsizes = [stepsizes for p in params]
        try:
            iter(momentum)
        except Exception:
            momentum = [momentum for p in params]
        if len(params) != len(grads):
            raise ValueError('params and grads have different lens')
        headings = [theano.shared(numpy.zeros_like(p.get_value(borrow=True)))
                for p in params]
        updates = []
        for s, p, gp, m, h in zip(stepsizes, params, grads, momentum,
                headings):
            updates.append((p, p + s * h))
            updates.append((h, m * h - (1.0 - m) * gp))
        return updates
