"""
Corruptor classes: classes that encapsulate the noise process for the DAE
training criterion.
"""
# Third-party imports
import numpy
import theano
from theano import tensor
T = tensor
from theano.printing import Print
# Shortcuts
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Corruptor(object):
    def __init__(self, corruption_level, rng=2001):
        """
        Allocate a corruptor object.

        Parameters
        ----------
        corruption_level : float
            Some measure of the amount of corruption to do. What this
            means will be implementation specific.
        rng : RandomState object or seed
            NumPy random number generator object (or seed for creating one)
            used to initialize a RandomStreams.
        """
        # The default rng should be build in a deterministic way
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
        self.corruption_level = corruption_level

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a noise process.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        else:
            return [self._corrupt(inp) for inp in inputs]

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs.

        Notes
        -----
        This is the method that all subclasses should implement. The logic in
        Corruptor.__call__ handles mapping over multiple tensor_like inputs.
        """
        raise NotImplementedError()

    def corruption_free_energy(self, corrupted_X, X):
        raise NotImplementedError()


class DummyCorruptor(Corruptor):
    def __call__(self, inputs):
        return inputs


class BinomialCorruptor(Corruptor):
    """
    A binomial corruptor sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """

    def _corrupt(self, x):
        """
        (Symbolically) corrupt the input with a binomial (masking) noise.

        Parameters
        ----------
        x : tensor_like,
            Theano symbolic representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted inputs,
            where individual inputs have been masked with independent
            probability equal to `self.corruption_level`.
        """
        return self.s_rng.binomial(
            size=x.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * x


class GaussianCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero
    mean isotropic Gaussian noise.
    """

    def __init__(self, stdev, rng=2001):
        super(GaussianCorruptor, self).__init__(corruption_level=stdev, rng=rng)

    def _corrupt(self, x):
        """
        (Symbolically) corrupt the inputs with Gaussian noise.

        Parameters
        ----------
        x : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs,
            where individual inputs have been corrupted by zero mean Gaussian
            noise with standard deviation equal to `self.corruption_level`.
        """
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        return noise + x

    def corruption_free_energy(self, corrupted_X, X):
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval

class SaltPepperCorruptor(Corruptor):
    """
    Corrupts the input with salt and pepper noise.

    Sets some elements of the tensor to 0 or 1.
    """
    def _corrupt(self, x):
        a = self.s_rng.binomial(
            size=x.shape,
            p=(1 - self.corruption_level),
            dtype=theano.config.floatX
        )

        b = self.s_rng.binomial(
            size=x.shape,
            p=0.5,
            dtype=theano.config.floatX
        )

        c = T.eq(a, 0) * b
        return x * a + c

class BinomialSampler(Corruptor):
    def __init__(self, *args, **kwargs):
        # pass up a 0 because corruption_level is not relevant here
        super(BinomialSampler, self).__init__(0, *args, **kwargs)

    def _corrupt(self, x):
        """
        Treats each element in matrix as probability for Bernoulli trial.

        Parameters
        ----------
        x : tensor_like
            A tensor like with all values between 0 and 1 (inclusive).

        Returns
        ----------
        y : tensor_like
            y_i,j is 1 with probability x_i,j.
        """

        return self.s_rng.binomial(size=x.shape, p=x,
                                   dtype=theano.config.floatX)

class MultinomialSampler(Corruptor):
    def __init__(self, *args, **kwargs):
        # corruption_level isn't relevant here
        super(MultinomialSampler, self).__init__(0, *args, **kwargs)

    def _corrupt(self, x):
        """
        Treats each row in matrix as a multinomial trial.

        Parameters
        ----------
        x : tensor_like
            x must be a matrix where all elements are non-negative (with at least
            one non-zero element)
        Returns
        ---------
        y : tensor_like
            y will have the same shape as x. Each row in y will be a one hot vector,
            and can be viewed as the outcome of the multinomial trial defined by
            the probabilities of that row in x.
        """
        normalized = x / x.sum(axis=1, keepdims=True)
        return self.s_rng.multinomial(pvals=normalized, dtype=theano.config.floatX)

class ComposedCorruptor(Corruptor):
    def __init__(self, *corruptors):
        """
        Parameters
        ----------
        corruptors : list of Corruptor objects
            The corruptors are applied in reverse order. This matches the typical
            function application notation. Thus ComposedCorruptor([a, b])._corrupt(X)
            is the same as a(b(X))

            Note
            ----
            Does NOT call Corruptor.__init__, so does not contain all of the
            standard fields for Corruptors.
        """
        # pass up the 0 for corruption_level (not relevant here)
        assert len(corruptors) >= 1
        self._corruptors = corruptors

    def _corrupt(self, x):
        result = x
        for c in reversed(self._corruptors):
            result = c(result)
        return result


##################################################
def get(str):
    """ Evaluate str into a corruptor object, if it exists """
    obj = globals()[str]
    if issubclass(obj, Corruptor):
        return obj
    else:
        raise NameError(str)
