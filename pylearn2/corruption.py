"""
Corruptor classes: classes that encapsulate the noise process for the DAE
training criterion.
"""
# Third-party imports
from __future__ import print_function

import numpy
import theano
from theano import tensor
T = tensor
from pylearn2.utils.rng import make_np_rng

# Shortcuts
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print('WARNING: using SLOW rng')
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

from pylearn2.expr.activations import rescaled_softmax

class Corruptor(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    corruption_level : float
        Some measure of the amount of corruption to do. What this means
        will be implementation specific.
    rng : RandomState object or seed, optional
        NumPy random number generator object (or seed for creating one)
        used to initialize a `RandomStreams`.
    """

    def __init__(self, corruption_level, rng=2001):
        # The default rng should be build in a deterministic way
        rng = make_np_rng(rng, which_method=['randn', 'randint'])
        seed = int(rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
        self.corruption_level = corruption_level

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a noise process.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of
            inputs to be corrupted, with the first dimension indexing
            training examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted
            inputs.
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
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.

        Notes
        -----
        This is the method that all subclasses should implement. The logic in
        Corruptor.__call__ handles mapping over multiple tensor_like inputs.
        """
        raise NotImplementedError()

    def corruption_free_energy(self, corrupted_X, X):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()


class DummyCorruptor(Corruptor):
    """
    .. todo::

        WRITEME
    """

    def __call__(self, inputs):
        """
        .. todo::

            WRITEME
        """
        return inputs


class BinomialCorruptor(Corruptor):
    """
    A binomial corruptor that sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        return self.s_rng.binomial(
            size=x.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * x


class DropoutCorruptor(BinomialCorruptor):
    """
    Sets inputs to 0 with probability of corruption_level and then
    divides by (1 - corruption_level) to keep expected activation
    constant.
    """

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        # for stability
        if self.corruption_level < 1e-5:
            return x

        dropped = super(DropoutCorruptor, self)._corrupt(x)
        return 1.0 / (1.0 - self.corruption_level) * dropped


class GaussianCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero mean isotropic
    Gaussian noise.

    Parameters
    ----------
    stdev : WRITEME
    rng : WRITEME
    """

    def __init__(self, stdev, rng=2001):
        super(GaussianCorruptor, self).__init__(corruption_level=stdev,
                                                rng=rng)

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        return noise + x

    def corruption_free_energy(self, corrupted_X, X):
        """
        .. todo::

            WRITEME
        """
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval


class SaltPepperCorruptor(Corruptor):
    """
    Corrupts the input with salt and pepper noise.

    Sets some elements of the tensor to 0 or 1. Only really makes sense
    to use on binary valued matrices.
    """
    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
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


class OneHotCorruptor(Corruptor):
    """
    Corrupts a one-hot vector by changing active element with some
    probability.
    """
    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        num_examples = x.shape[0]
        num_classes = x.shape[1]

        keep_mask = T.addbroadcast(
            self.s_rng.binomial(
                size=(num_examples, 1),
                p=1 - self.corruption_level,
                dtype='int8'
            ),
            1
        )

        # generate random one-hot matrix
        pvals = T.alloc(1.0 / num_classes, num_classes)
        one_hot = self.s_rng.multinomial(size=(num_examples,), pvals=pvals)

        return keep_mask * x + (1 - keep_mask) * one_hot

class SmoothOneHotCorruptor(Corruptor):
    """
    Corrupts a one-hot vector in a way that preserves some information.

    This adds Gaussian noise to a vector and then computes the softmax.
    """

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        return rescaled_softmax(x + noise)


class BinomialSampler(Corruptor):
    """
    .. todo::

        WRITEME
    """

    def __init__(self, *args, **kwargs):
        # pass up a 0 because corruption_level is not relevant here
        super(BinomialSampler, self).__init__(0, *args, **kwargs)

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        return self.s_rng.binomial(size=x.shape, p=x,
                                   dtype=theano.config.floatX)


class MultinomialSampler(Corruptor):
    """
    .. todo::

        WRITEME
    """

    def __init__(self, *args, **kwargs):
        # corruption_level isn't relevant here
        super(MultinomialSampler, self).__init__(0, *args, **kwargs)

    def _corrupt(self, x):
        """
        Treats each row in matrix as a multinomial trial.

        Parameters
        ----------
        x : tensor_like
            x must be a matrix where all elements are non-negative
            (with at least one non-zero element)
        Returns
        -------
        y : tensor_like
            y will have the same shape as x. Each row in y will be a
            one hot vector, and can be viewed as the outcome of the
            multinomial trial defined by the probabilities of that row
            in x.
        """
        normalized = x / x.sum(axis=1, keepdims=True)
        return self.s_rng.multinomial(pvals=normalized, dtype=theano.config.floatX)


class ComposedCorruptor(Corruptor):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    corruptors : list of Corruptor objects
        The corruptors are applied in reverse order. This matches the
        typical function application notation. Thus
        `ComposedCorruptor(a, b)._corrupt(X)` is the same as `a(b(X))`.

    Notes
    -----
    Does NOT call Corruptor.__init__, so does not contain all of the
    standard fields for Corruptors.
    """

    def __init__(self, *corruptors):
        # pass up the 0 for corruption_level (not relevant here)
        assert len(corruptors) >= 1
        self._corruptors = corruptors

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
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
