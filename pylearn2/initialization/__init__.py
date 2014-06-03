"""
Objects for encapsulating parameter initialization strategies.
"""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"

import numpy as np
import theano
from pylearn2.utils import wraps


# TODO: To add --
#  - object representing initialization from dataset marginals
#  - a GlorotBengio object implementing Glorot & Bengio (2010)'s strategy
#    for tanh nets.


class NdarrayInitialization(object):
    """
    Base class specifying the interface for these objects.
    """
    def initialize(self, rng, shape, atom_axis=-1, fan_in=None, fan_out=None,
                   *args, **kwargs):
        """
        Generate an initial set of parameters from a given
        distribution. This should generally be called by the model,
        not directly by the user.

        Parameters
        ----------
        rng : object
            A `numpy.random.RandomState`.
        shape : tuple
            A shape tuple for the requested parameter array shape.
        atom_axis : int, optional
            The axis of `shape` corresponding to the number
            hidden units/dictionary elements/etc. By default, use -1
            (i.e. the last axis).
        fan_in : int, optional
            The number of units in a neural network with output that
            feeds into each unit in this layer. Certain kinds of
            initialization strategies use this to calculate properties
            of the distribution used for random generation. By default
            it can be calculated as the product of all elements of
            `shape` except the one in the position corresponding to
            `atom_axis`.
        fan_out : int, optional
            The number of units in a neural network which take input
            from each unit in this layer. Certain kinds of initialization
            strategies use this to calculate properties of the
            distribution used for random generation. As it is not
            possible to infer this quantity from `shape`, it must be
            provided by the model if using an initialization strategy
            which requires it.

        Returns
        -------
        initialized : ndarray
            An ndarray with values drawn from the distribution
            specified by this object, of shape `shape`, with dtype
            `theano.config.floatX`.
        """
        raise NotImplementedError("Instantiate a subclass of %s, not %s" %
                                  ((self.__class__.__name__,) * 2))


class Constant(object):
    """
    Initialize parameters to a constant. The constant may be a scalar
    or an array_like of any shape that is broadcastable with the requested
    parameter arrays.

    Parameters
    ----------
    constant : array_like
        The initialization value to use. Must be a scalar or an
        ndarray (or compatible object, such as a nested list) that
        has a shape that is broadcastable with any shape requested
        by `initialize`.
    """
    def __init__(self, constant):
        self._constant = np.asarray(constant)

    @wraps(NdarrayInitialization.initialize)
    def initialize(self, rng, shape, *args, **kwargs):
        dest = np.empty(shape, dtype=theano.config.floatX)
        try:
            np.broadcast(dest, self._constant)
        except ValueError:
            raise ValueError("Constant initialization failed: needed shape "
                             "but got non-braodcastable shape %s" %
                             (str(shape), self._constant.shape))
        dest[...] = self._constant
        return dest

    def __str__(self):
        param = str(self._constant)
        if len(param) > 20:
            param = "...<str too long>..."
        return "Constant(%s)" % param


class IsotropicGaussian(NdarrayInitialization):
    """
    Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults
        to 1.
    """
    def __init__(self, mean=0, std=1):
        self._mean = mean
        self._std = std

    @wraps(NdarrayInitialization.initialize)
    def initialize(self, rng, shape, *args, **kwargs):
        m = rng.normal(self._mean, self._std, size=shape)
        return m.astype(theano.config.floatX)

    def __str__(self):
        return ("IsotropicGaussian(mean=%s, std=%s)" %
                (self._mean, self._std))


class Uniform(NdarrayInitialization):
    """
    Initialize parameters from a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        The mean of the uniform distribution (i.e. the center
        of mass for the density function); Defaults to 0.

    width : float, optional
        One way of specifying the range of the uniform
        distribution. The support will be [mean - width/2,
        mean + width/2]. **Exactly one** of `width` or `std`
        must be specified.

    std : float, optional
        An alternative method of specifying the range of the
        uniform distribution. Chooses the width of the uniform
        such that random variates will have a desired standard
        deviation. **Exactly one** of `width` or `std` must
        be specified.

    """
    def __init__(self, mean=0., width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self._width = np.sqrt(12) * std
            self._std = std  # For display purposes.
        else:
            self._width = width
            self._std = None  # For display purposes.
        self._mean = mean

    @wraps(NdarrayInitialization.initialize)
    def initialize(self, rng, shape, *args, **kwargs):
        w = self._width / 2
        m = rng.uniform(self._mean - w, self._mean + w, size=shape)
        return m.astype(theano.config.floatX)

    def __str__(self):
        pairs = []
        if self._mean != 0:
            pairs.append(('mean', '%s' % (self._mean,)))
        if self._std is not None:
            pairs.append(('std', '%s' % (self._std,)))
        else:
            pairs.append(('width', '%s' % (self._width,)))
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, ', '.join('='.join(p) for p in pairs))


class SparseInitialization(NdarrayInitialization):
    def __init__(self, base_initialization, prob_nonzero=None,
                 num_nonzero=None):
        """
        Initialize a parameter array in a sparse fashion, with
        a certain randomly chosen fraction of elements initialized
        from `base_initialization`, and the rest initialized to 0.

        Parameters
        ----------
        base_initialization : object
            Delegated object specifying the distribution/strategy to
            use for the non-zero parameter.

        prob_nonzero : float, optional
            If specified, each parameter element will be non-zero
            (and sampled from the base initialization strategy)
            with independent probability `prob_nonzero`. **Exactly
            one** of `prob_nonzero` or `num_nonzero` must be specified.

        num_nonzero : int, optional
            If specified, exactly this many weights per atom (filter)
            will be sampled from base_initialization. Mutually
            exclusive with num_nonzero. **Exactly one** of `prob_nonzero`
            or `num_nonzero` must be speciifed.

        Notes
        -----
        TODO: Add Martens (2010), Sutskever (2013) reference.
        """
        if (num_nonzero is not None) == (prob_nonzero is not None):
            raise ValueError("must specify num_nonzero or prob_nonzero, "
                             "but not both")
        elif prob_nonzero is not None and not (0 < float(prob_nonzero) <= 1):
            raise ValueError("prob_nonzero must be between 0 and 1")
        if num_nonzero is not None and num_nonzero != int(num_nonzero):
            raise ValueError("num_nonzero must be integer")
        self._prob_nonzero = (float(prob_nonzero) if prob_nonzero is not None
                              else None)
        self._num_nonzero = (int(num_nonzero) if num_nonzero is not None
                             else None)
        self._base = base_initialization

    @wraps(NdarrayInitialization.initialize)
    def initialize(self, rng, shape, atom_axis=-1, *args, **kwargs):
        if atom_axis < 0:
            atom_axis += len(shape)
        values = self._base.initialize(rng, shape, atom_axis, *args, **kwargs)
        if self._prob_nonzero is not None:
            values *= (rng.uniform(size=shape) < self._prob_nonzero)
        elif self._num_nonzero is not None:
            nnz = self._num_nonzero
            per_atom = values.size // shape[atom_axis]
            if nnz > per_atom:
                raise ValueError("%d non-zero elements per atom requested "
                                 "but only %d parameters per atom: shape %s, "
                                 "atom_axis=%d" % (nnz, per_atom, shape,
                                                   atom_axis))

            mask = np.zeros((shape[atom_axis], per_atom),
                            dtype=np.int8)
            indices = np.arange(per_atom)
            # TODO: use something else to generate this structured random
            # matrix. There's got to be a cleverer algorithm for this but
            # I'm not seeing it.
            for i in xrange(shape[atom_axis]):
                rng.shuffle(indices)
                mask[i, indices[:nnz]] = 1
            # Swap the atom axis into position 0, save this shape.
            values = values.swapaxes(0, atom_axis)
            vshape = values.shape
            # Reshape into a rectangle.
            values = values.reshape((values.shape[0], -1))
            values *= mask
            # Undo the swapping and the reshape.
            values = values.reshape(vshape).swapaxes(atom_axis, 0)
        return values

    def __str__(self):
        base = str(self._base)
        arg_str = ('prob_nonzero' if self._prob_nonzero is not None
                   else 'num_nonzero')
        arg = (('%s' % self._prob_nonzero) if self._prob_nonzero is not None
               else ('%d' % self._num_nonzero))
        class_name = self.__class__.__name__
        return '%s(base=%s, %s=%s)' % (class_name, base, arg_str, arg)
