"""Generic "model" class."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import warnings

import numpy as np

from theano.compat.python2x import OrderedDict
from theano import tensor as T
from theano import shared

from pylearn2.space import NullSpace


class Model(object):
    """
    A class representing a model with learnable parameters.
    """

    def get_default_cost(self):
        """
        Returns the default cost to use with this model.
        """

        raise NotImplementedError(str(type(self))+ " does not implement get_default_cost.")

    def train_all(self, dataset):
        """
        If implemented, performs one epoch of training.
        This method is useful for models with highly specialized training
        algorithms for which is does not make much sense to factor the training
        code into a separate class. It is also useful for implementors that want
        to make their model trainable without enforcing compatibility with
        pylearn2 TrainingAlgorithms.

        Parameters
        ----------
        dataset: pylearn2.datasets.dataset.Dataset
            Dataset object to draw training data from
        """
        raise NotImplementedError(str(type(self))+" does not implement train_all.")

    def continue_learning(self):
        """
        If train_all is used to train the model, this method is used to determine
        when the training process has converged. This method is called after the
        monitor has been run on the latest parameters.

        Returns
        -------
        rval : bool
            True if training should continue
        """

        raise NotImplementedError(str(type(self))+" does not implement continue_learning.")


    def train_batch(self, dataset, batch_size):
        """
        If implemented, performs an update on a single minibatch.

        Parameters
        ----------
        dataset: pylearn2.datasets.dataset.Dataset
                The object to draw training data from.
        batch_size: int
                Size of the minibatch to draw from dataset.

        Returns
        -------
        rval : bool
            True if the method should be called again for another update. \
            False if convergence has been reached.
        """
        raise NotImplementedError()

    def get_weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement get_weights_view_shape (perhaps by design)")

    def get_monitoring_channels(self, data):
        """
        Get monitoring channels for this model.

        Parameters
        ----------
        data: tensor_like, or (possibly nested) tuple of tensor_likes,
            This is data on which the monitoring quantities will be \
            calculated (e.g., a validation set). See \
            `self.get_monitoring_data_specs()`.

        Returns
        -------
        channels : dict
            A dictionary with strings as keys, mapping channel names to \
            symbolic values that depend on the variables in `data`.

        Notes
        -----
        You can make any channel names you want, just try to make sure they
        won't collide with names made by the training Cost, etc. Anything you
        think is worth monitoring during training can be added here. You
        probably want to control which channels get added with some config
        option for your model.
        """
        space, source = self.get_monitoring_data_specs()
        space.validate(data)
        return OrderedDict()

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channels.

        This implementation returns an empty data_specs, appropriate for
        when no monitoring channels are defined, or when none of the channels
        actually need data (for instance, if they only monitor functions
        of the model's parameters).
        """
        return (NullSpace(), '')

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        pass

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """

        raise NotImplementedError(str(type(self))+" does not implement get_weights (perhaps by design)")

    def get_weights_topo(self):
        """
        .. todo::

            WRITEME
        """

        raise NotImplementedError(str(type(self))+" does not implement get_weights_topo (perhaps by design)")


    def score(self, V):
        """
        Compute a "score function" for this model, if this model has
        probabilistic semantics.

        Parameters
        ----------
        V : tensor_like, 2-dimensional
            A batch of i.i.d. examples with examples indexed along the \
            first axis and features along the second. This is data on which \
            the monitoring quantities will be calculated (e.g., a validation \
            set).

        Returns
        -------
        score : tensor_like
            The gradient of the negative log probability of the model \
            on the given datal.

        Notes
        -----
        If the model implements a probability distribution on R^n,
        this method should return the gradient of the log probability
        of the batch with respect to V, or raise an exception explaining
        why this is not possible.
        """
        return T.grad(-self.free_energy(V).sum(), V)

    def get_lr_scalers(self):
        """
        .. todo::

            WRITEME
        """
        return OrderedDict()

    def censor_updates(self, updates):
        """
        This method should check all updates that act on shared variables
        held by the model and make sure they are valid. For example, if
        a given hyperparameter is not meant to be learned, censor_updates
        should remove it from the dictionary. If a parameter has a restricted
        range, e.g.. if it is the precision of a normal distribution,
        censor_updates should clip its update to that range. If a parameter
        has any other special properties, its updates should be modified
        to respect that here, e.g. a matrix that must be orthogonal should
        have its update value modified to be orthogonal here.

        This is the main mechanism used to make sure that generic training
        algorithms such as those found in pylearn2.training_algorithms
        respect the specific properties of the models passed to them.

        Parameters
        ----------
        updates : dict
            A dictionary mapping shared variables to symbolic values they \
            will be updated to

        Returns
        -------
        WRITEME
        """

        pass

    def get_input_space(self):
        """
        Returns an instance of pylearn2.space.Space describing the format of
        the vector space that the model operates on (this is a generalization
        of get_input_dim)
        """

        return self.input_space

    def get_output_space(self):
        """
        Returns an instance of pylearn2.space.Space describing the format of
        the vector space that the model outputs (this is a generalization
        of get_output_dim)
        """

        return self.output_space

    def get_input_source(self):
        """
        Returns a string, stating the source for the input. By default the
        input source (when is the only one) is called 'features'.
        """
        return 'features'

    def get_target_source(self):
        """
        Returns a string, stating the source for the output. By default the
        output source (when is the only one) is called 'targets'.
        """
        return 'targets'

    def free_energy(self, V):
        """
        Compute the free energy of data examples, if this model has
        probabilistic semantics.

        Parameters
        ----------
        V : tensor_like, 2-dimensional
            A batch of i.i.d. examples with examples indexed along the \
            first axis and features along the second. This is data on which \
            the monitoring quantities will be calculated (e.g., a validation \
            set).

        Returns
        -------
        free_energy : tensor, 1-dimensional
            A (symbolic) vector of free energies for each data example in \
            `V`, i.e.  `free_energy[i] = F(V[i])`.
        """
        raise NotImplementedError()

    def get_params(self):
        """
        Returns the parameters that define the model.

        Returns
        -------
        params : list
            A list of (Theano shared variable) parameters of the model.

        Notes
        -----
        By default, this returns a copy of the _params attribute, which
        individual models can simply fill with the list of model parameters.
        Alternatively, models may override `get_params`, so this should
        be considered the public interface to model parameters -- directly
        accessing or modifying _params is at-your-own-risk, as it may
        or may not exist.

        This is the main mechanism by which generic training algorithms
        like SGD know which values to update, however, even model
        parameters that should not be learned ought to be included here,
        so that the model's parameter set is more predictable.

        Parameters may be included here but held constant during
        learning via the `censor_updates` method.
        """
        return list(self._params)

    def get_param_values(self, borrow=False):
        """
        Returns numerical values for the parameters that define the model.

        Parameters
        ----------
        borrow : bool
            Flag to be passed to the `.get_value()` method of the \
            shared variable. If `False`, a copy will always be returned.

        Returns
        -------
        params : list
            A list of `numpy.ndarray` objects containing the current \
            parameters of the model.

        Notes
        -----
        This is the main  mechanism by which generic training algorithms
        like SGD know which values to update, however, even model
        parameters that should not be learned ought to be included here,
        so that the model's parameter set is more predictable.

        Parameters may be included here but held constant during
        learning via the `censor_updates` method.
        """
        # I think there's a bug here, because get_params returns a set
        # but sets have no defined iteration order, so get_param_values
        # might return things in one order and set_param_values might
        # try to put them back in in a different order
        assert not isinstance(self.get_params(), set)
        return [param.get_value(borrow=borrow) for param in self.get_params()]

    def set_param_values(self, values, borrow=False):
        """
        .. todo::

            WRITEME properly

        Sets the values of the parameters that define the model
        """
        for param, value in zip(self.get_params(), values):
            param.set_value(value, borrow=borrow)

    def redo_theano(self):
        """
        Re-compiles all Theano functions used internally by the model.
        This function is often called after a model is unpickled from
        disk, since Theano functions are not pickled. However, it is
        not always called. This allows scripts like show_weights.py
        to rapidly unpickle a model and inspect its weights without
        needing to recompile all of its learning machinery.

        All Theano functions compiled by this method should be registered
        with the register_names_to_del method.
        """
        pass

    def get_input_dim(self):
        """
        Returns the number of visible units of the model.
        Deprecated; this assumes the model operates on a vector.
        Use get_input_space instead.
        """
        raise NotImplementedError()

    def get_output_dim(self):
        """
        Returns the number of visible units of the model.
        Deprecated; this assumes the model operates on a vector.
        Use get_input_space instead.
        """
        raise NotImplementedError()

    def __getstate__(self):
        """
        This is the method that pickle/cPickle uses to determine what
        portion of the model to serialize. We remove all fields listed in
        `self.fields_to_del`. In particular, this should include all Theano
        functions, since they do not play nice with pickling.
        """
        d = OrderedDict()
        names_to_del = getattr(self, 'names_to_del', set())
        names_to_keep = set(self.__dict__.keys()).difference(names_to_del)
        for name in names_to_keep:
            d[name] = self.__dict__[name]
        return d

    def __setstate__(self, d):
        """
        .. todo::

            WRITEME
        """
        self.__dict__.update(d)

    def __init__(self):
        """
        .. todo::

            WRITEME
        """
        self.names_to_del = set()
        self._test_batch_size = 2

    def get_test_batch_size(self):
        """
        Batches of examples used to initialize X.tag.test_value should have this
        many examples if used as input to the model.  (The model
        specifies the number of examples in case it needs a fixed batch
        size or to keep the memory usage of testing under control.)
        """
        return self._test_batch_size

    def register_names_to_del(self, names):
        """
        Register names of fields that should not be pickled.

        Parameters
        ----------
        names : iterable
            A collection of strings indicating names of fields on this \
            object that should not be pickled.

        Notes
        -----
        All names registered will be deleted from the dictionary returned
        by the model's `__getstate__` method (unless a particular model
        overrides this method).
        """
        if isinstance(names, basestring):
            names = [names]
        try:
            assert all(isinstance(n, basestring) for n in iter(names))
        except (TypeError, AssertionError):
            raise ValueError('Invalid names argument')
        self.names_to_del = self.names_to_del.union(names)

