"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
N = numpy
np = numpy
import theano
from theano import tensor
T = tensor
from theano.tensor import nnet

# Local imports
from pylearn2.base import Block, StackedBlocks
from pylearn2.utils import as_floatX, safe_update, sharedX
from pylearn2.models import Model
from pylearn2.optimizer import SGDOptimizer
from pylearn2.expr.basic import theano_norms
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.space import VectorSpace
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
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
    rbm : object
        An instance of `RBM` or a derived class, or one implementing
        the RBM interface.
    sampler : object
        An instance of `Sampler` or a derived class, or one implementing
        the sampler interface.
    optimizer : object
        An instance of `Optimizer` or a derived class, or one implementing
        the optimizer interface (typically an `SGDOptimizer`).
    """
    pos_v = visible_batch
    neg_v = sampler.particles
    grads = model.ml_gradients(pos_v, neg_v)
    ups = optimizer.updates(gradients=grads)

    # Add the sampler's updates (negative phase particles, etc.).
    safe_update(ups, sampler.updates())
    return ups


class Sampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.
    """
    def __init__(self, rbm, particles, rng):
        """
        Construct a Sampler.

        Parameters
        ----------
        rbm : object
            An instance of `RBM` or a derived class, or one implementing
            the `gibbs_step_for_v` interface.
        particles : ndarray
            An initial state for the set of persistent Narkov chain particles
            that will be updated at every step of learning.
        rng : RandomState object
            NumPy random number generator object used to initialize a
            RandomStreams object used in training.
        """
        self.__dict__.update(rbm=rbm)
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
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
    """
    def __init__(self, rbm, particles, rng, steps=1, particles_clip=None):
        """
        Construct a BlockGibbsSampler.

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
        particles_clip: None or (min, max) pair
            The values of the returned particles will be clipped between
            min and max.
        """
        super(BlockGibbsSampler, self).__init__(rbm, particles, rng)
        self.steps = steps
        self.particles_clip = particles_clip

    def updates(self, particles_clip=None):
        """
        Get the dictionary of updates for the sampler's persistent state
        at each step..

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

    """
    def __init__(self, nvis = None, nhid = None,
            vis_space = None,
            hid_space = None,
            transformer = None,
            irange=0.5, rng=None, init_bias_vis = None,
            init_bias_vis_marginals = None, init_bias_hid=0.0,
            base_lr = 1e-3, anneal_start = None, nchains = 100, sml_gibbs_steps = 1,
            random_patches_src = None,
            monitor_reconstruction = False):

        """
        Construct an RBM object.

        Parameters
        ----------
        nvis : int
            Number of visible units in the model.
            (Specifying this implies that the model acts on a vector,
            i.e. it sets vis_space = pylearn2.space.VectorSpace(nvis) )
        nhid : int
            Number of hidden units in the model.
            (Specifying this implies that the model acts on a vector)
        vis_space:
            A pylearn2.space.Space object describing what kind of vector
            space the RBM acts on. Don't specify if you used nvis / hid
        hid_space:
            A pylearn2.space.Space object describing what kind of vector
            space the RBM's hidden units live in. Don't specify if you used
            nvis / nhid
        init_bias_vis_marginals: either None, or a Dataset to use to initialize
            the visible biases to the inverse sigmoid of the data marginals
        irange : float, optional
            The size of the initial interval around 0 for weights.
        rng : RandomState object or seed
            NumPy RandomState object to use when initializing parameters
            of the model, or (integer) seed to use to create one.
        init_bias_vis : array_like, optional
            Initial value of the visible biases, broadcasted as necessary.
        init_bias_hid : array_like, optional
            initial value of the hidden biases, broadcasted as necessary.
        monitor_reconstruction : if True, will request a monitoring channel to monitor
            reconstruction error
        random_patches_src: Either None, or a Dataset from which to draw random patches
            in order to initialize the weights. Patches will be multiplied by irange

        Parameters for default SML learning rule:

            base_lr : the base learning rate
            anneal_start : number of steps after which to start annealing on a 1/t schedule
            nchains: number of negative chains
            sml_gibbs_steps: number of gibbs steps to take per update

        """

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

        if rng is None:
            # TODO: global rng configuration stuff.
            rng = numpy.random.RandomState(1001)
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
                        #assert type(irange) == type(0.01)
                        #assert irange == 0.01
                        W = irange * random_patches_src.get_batch_design(nhid).T

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
            raise ValueError("bad shape or value for init_bias_vis")
        self.bias_vis = sharedX(b_vis, name='bias_vis', borrow=True)

        try:
            b_hid = self.hid_space.get_origin()
            b_hid += init_bias_hid
        except ValueError:
            raise ValueError('bad shape or value for init_bias_hid')
        self.bias_hid = sharedX(b_hid, name='bias_hid', borrow=True)

        self.random_patches_src = random_patches_src
        self.register_names_to_del(['random_patches_src'])


        self.__dict__.update(nhid=nhid, nvis=nvis)
        self._params = list(self.transformer.get_params().union([self.bias_vis, self.bias_hid]))

        self.base_lr = base_lr
        self.anneal_start = anneal_start
        self.nchains = nchains
        self.sml_gibbs_steps = sml_gibbs_steps

    def get_input_dim(self):
        if not isinstance(self.vis_space, VectorSpace):
            raise TypeError("Can't describe "+str(type(self.vis_space))+" as a dimensionality number.")
        return self.vis_space.dim

    def get_output_dim(self):
        if not isinstance(self.hid_space, VectorSpace):
            raise TypeError("Can't describe "+str(type(self.hid_space))+" as a dimensionality number.")
        return self.hid_space.dim

    def get_input_space(self):
        return self.vis_space

    def get_output_space(self):
        return self.hid_space

    def get_params(self):
        return [param for param in self._params]

    def get_weights(self, borrow=False):

        weights ,= self.transformer.get_params()

        return weights.get_value(borrow=borrow)

    def get_weights_topo(self, borrow=False):
        return self.transformer.get_weights_topo(borrow = borrow)

    def get_weights_format(self):
        return ['v', 'h']


    def get_monitoring_channels(self, V, Y = None):

        theano_rng = RandomStreams(42)

        #TODO: re-enable this in the case where self.transformer
        #is a matrix multiply
        #norms = theano_norms(self.weights)

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
                 #'W_min' : T.min(self.weights),
                 #'W_max' : T.max(self.weights),
                 #'W_norms_min' : T.min(norms),
                 #'W_norms_max' : T.max(norms),
                 #'W_norms_mean' : T.mean(norms),
                'reconstruction_error' : self.reconstruction_error(V, theano_rng) }

    def ml_gradients(self, pos_v, neg_v):
        """
        Get the contrastive gradients given positive and negative phase
        visible units.

        Parameters
        ----------
        pos_v : tensor_like
            Theano symbolic representing a minibatch on the visible units,
            with the first dimension indexing training examples and the second
            indexing data dimensions (usually actual training data).
        neg_v : tensor_like
            Theano symbolic representing a minibatch on the visible units,
            with the first dimension indexing training examples and the second
            indexing data dimensions (usually reconstructions of the data or
            sampler particles from a persistent Markov chain).

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


    def learn(self, dataset, batch_size):
        """ A default learning rule based on SML """
        self.learn_mini_batch(dataset.get_batch_design(batch_size))

    def learn_mini_batch(self, X):
        """ A default learning rule based on SML """

        if not hasattr(self, 'learn_func'):
            self.redo_theano()

        rval =  self.learn_func(X)

        return rval

    def redo_theano(self):
        """ Compiles the theano function for the default learning rule """

        init_names = dir(self)

        minibatch = tensor.matrix()

        optimizer = SGDOptimizer(self, self.base_lr, self.anneal_start)

        sampler = sampler = BlockGibbsSampler(self, 0.5 + np.zeros((self.nchains, self.get_input_dim())), self.rng,
                                                  steps= self.sml_gibbs_steps)


        updates = training_updates(visible_batch=minibatch, model=self,
                                            sampler=sampler, optimizer=optimizer)

        self.learn_func = theano.function([minibatch], updates=updates)

        final_names = dir(self)

        self.register_names_to_del([name for name in final_names if name not in init_names])

    def gibbs_step_for_v(self, v, rng):
        """
        Do a round of block Gibbs sampling given visible configuration

        Parameters
        ----------
        v  : tensor_like
            Theano symbolic representing the hidden unit states for a batch of
            training examples (or negative phase particles), with the first
            dimension indexing training examples and the second indexing data
            dimensions.
        rng : RandomStreams object
            Random number generator to use for sampling the hidden and visible
            units.

        Returns
        -------
        v_sample : tensor_like
            Theano symbolic representing the new visible unit state after one
            round of Gibbs sampling.
        locals : dict
            Contains the following auxiliary state as keys (all symbolics
            except shape tuples):
             * `h_mean`: the returned value from `mean_h_given_v`
             * `h_mean_shape`: shape tuple indicating the size of `h_mean` and
               `h_sample`
             * `h_sample`: the stochastically sampled hidden units
             * `v_mean_shape`: shape tuple indicating the shape of `v_mean` and
               `v_sample`
             * `v_mean`: the returned value from `mean_v_given_h`
             * `v_sample`: the stochastically sampled visible units
        """
        h_mean = self.mean_h_given_v(v)
        assert h_mean.type.dtype == v.type.dtype
        # For binary hidden units
        # TODO: factor further to extend to other kinds of hidden units
        #       (e.g. spike-and-slab)
        h_sample = rng.binomial(size = h_mean.shape, n = 1 , p = h_mean, dtype=h_mean.type.dtype)
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
        v  : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the one or several
            minibatches on the visible units, with the first dimension indexing
            training examples and the second indexing data dimensions.

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
        h  : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the one or several
            minibatches on the hidden units, with the first dimension indexing
            training examples and the second indexing data dimensions.

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

    def mean_h_given_v(self, v):
        """
        Compute the mean activation of the hidden units given visible unit
        configurations for a set of training examples.

        Parameters
        ----------
        v  : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the hidden unit
            states for a batch (or several) of training examples, with the
            first dimension indexing training examples and the second indexing
            data dimensions.

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
        h  : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the hidden unit
            states for a batch (or several) of training examples, with the
            first dimension indexing training examples and the second indexing
            hidden units.

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
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
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
        across a minibatch after a Gibbs
        step starting from the training data.

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.
        rng : RandomStreams object
            Random number generator to use for sampling the hidden and visible
            units.

        Returns
        -------
        mse : tensor_like
            0-dimensional tensor (essentially a scalar) indicating the mean
            reconstruction error across the minibatch.

        Notes
        -----
        The reconstruction used to assess error samples only the hidden
        units. For the visible units, it uses the conditional mean.
        No sampling of the visible units is done, to reduce noise in the estimate.
        """
        sample, _locals = self.gibbs_step_for_v(v, rng)
        return ((_locals['v_mean'] - v) ** 2).sum(axis=1).mean()


class GaussianBinaryRBM(RBM):
    """
    An RBM with Gaussian visible units and binary hidden units.
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
        """
        Allocate a GaussianBinaryRBM object.

        Parameters
        ----------
        nvis : int
            Number of visible units in the model.
        nhid : int
            Number of hidden units in the model.
        energy_function_class:
            TODO: finish comment
        irange : float, optional
            The size of the initial interval around 0 for weights.
        rng : RandomState object or seed
            NumPy RandomState object to use when initializing parameters
            of the model, or (integer) seed to use to create one.
        mean_vis : bool, optional
            Don't actually sample visibles; make sample method simply return
            mean.
        init_sigma : Initial value of the sigma variable.
                    If init_sigma is a scalar and sigma is not, will be broadcasted
        min_sigma, max_sigma: elements of sigma are clipped to this range during learning
        init_bias_hid : scalar or 1-d array of length `nhid`
            Initial value for the biases on hidden units.
        """
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

    def censor_updates(self, updates):
        if self.sigma_driver in updates:
            assert self.learn_sigma
            updates[self.sigma_driver] = T.clip(
                updates[self.sigma_driver],
                self.min_sigma / self.sigma_lr_scale,
                self.max_sigma / self.sigma_lr_scale
            )

    def score(self, V):
        return self.energy_function.score(V)

    def P_H_given_V(self, V):
        return self.energy_function.P_H_given(V)

    def mean_v_given_h(self, h):
        """
        Compute the mean activation of the visibles given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        h  : tensor_like
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
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
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        f : tensor_like
            1-dimensional tensor representing the
            free energy of the visible unit configuration
            for each example in the batch
        """

        """hid_inp = self.input_to_h_from_v(v)
        squared_term = ((self.bias_vis - v) ** 2.) / (2. * self.sigma)
        rval =  squared_term.sum(axis=1) - nnet.softplus(hid_inp).sum(axis=1)
        assert len(rval.type.broadcastable) == 1"""

        return self.energy_function.free_energy(V)

    def free_energy(self, V):
        return self.energy_function.free_energy(V)
    #

    def sample_visibles(self, params, shape, rng):
        """
        Stochastically sample the visible units given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        params : list
            List of the necessary parameters to sample :math:`p(v|h)`. In the
            case of a Gaussian-binary RBM this is a single-element list
            containing the conditional mean.

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing stochastic samples from :math:`p(v|h)`

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
    TODO: reformat doc

    alpha    : vector of length nslab, diagonal precision term on s.
    b        : vector of length nhid, hidden unit bias.
    B        : vector of length nvis, diagonal precision on v.
               Lambda in ICML2011 paper.
    Lambda   : matrix of shape nvis x nhid, whose i-th column encodes a
               diagonal precision on v, conditioned on h_i.
               phi in ICML2011 paper.
    log_alpha: vector of length nslab, precision on s.
    mu       : vector of length nslab, mean parameter on s.
    W        : matrix of shape nvis x nslab, weights of the nslab linear
               filters s.
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
        if rng is None:
            # TODO: global rng default seed
            rng = numpy.random.RandomState(1001)

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
        raise NotImplementedError('mu_pooled_ssRBM.sample_visibles')

    def input_to_h_from_v(self, v):
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
        v_var = 1 / (self.B + tensor.dot(h, self.Lambda.T))
        s3 = s.reshape((
                -1,
                self.nhid,
                self.n_s_per_h))
        hs = h.dimshuffle(0, 1, 'x') * s3
        v_mu = tensor.dot(hs.flatten(2), self.W.T) * v_var
        return v_mu, v_var

    def mean_var_s_given_v_h1(self, v):
        alpha = self.alpha
        return (self.mu + tensor.dot(v, self.W) / alpha,
                1.0 / alpha)

    ## TODO?
    def mean_v_given_h(self, h):
        raise NotImplementedError('mu_pooled_ssRBM.mean_v_given_h')

    def free_energy_given_v(self, v):
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
