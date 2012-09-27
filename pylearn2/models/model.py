"""Generic "model" class."""
from theano import tensor as T
from theano import shared
import numpy as np
import warnings


class Model(object):
    def train_all(self, dataset):
        """
        If implemented, performs one epoch of training.

        Parameters
        ----------
        dataset: The pylearn2.datasets.dataset.Dataset object to draw training
                data from

        Return value:
            True if the method should be called again for another epoch
            False if convergence has been reached
        """
        raise NotImplementedError()

    def train_batch(self, dataset, batch_size):
        """
        If implemented, performs an update on a single minibatch.

        Parameters
        ----------
        dataset: pylearn2.datasets.dataset.Dataset 
                The object to draw training data from.
        batch_size: integer
                Size of the minibatch to draw from dataset.

        Return value:
            True if the method should be called again for another update.
            False if convergence has been reached.
        """
        raise NotImplementedError()

    def get_monitoring_channels(self, V, Y = None):
        """
        Get monitoring channels for this model.

        Parameters
        ----------
        V : tensor_like, 2-dimensional
            A batch of i.i.d. examples with examples indexed along the
            first axis and features along the second. This is data on which
            the monitoring quantities will be calculated (e.g., a validation
            set).
        Y : optional class labels. Usually I have been representing them as
            a one-hot design matrix but we don't really have a standard yet.

        Returns
        -------
        channels : dict
            A dictionary with strings as keys, mapping channel names to
            symbolic values that depend on V.

        Notes
        -----
        You can make any channel names you want, just try to make sure they
        won't collide with names made by the training Cost, etc. Anything you
        think is worth monitoring during training can be added here. You
        probably want to control which channels get added with some config
        option for your model.
        """
        return {}

    def set_batch_size(self, batch_size):
        pass

    def score(self, V):
        """
        Compute a "score function" for this model, if this model has
        probabilistic semantics.

        Parameters
        ----------
        V : tensor_like, 2-dimensional
            A batch of i.i.d. examples with examples indexed along the
            first axis and features along the second. This is data on which
            the monitoring quantities will be calculated (e.g., a validation
            set).

        Returns
        -------
        score : tensor_like
            The gradient of the negative log probability of the model
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
        return {}

    def censor_updates(self, updates):
        """
        updates: a dictionary mapping shared variables to symbolic values
                they will be updated to

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
        respect the specific properties of the models passed to them."""
        pass

    def get_input_space(self):
        """ Returns an instance of pylearn2.space.Space describing
        the format of the vector space that the model operates oni
        (this is a generalization of get_input_dim) """

        return self.input_space

    def get_output_space(self):
        """ Returns an instance of pylearn2.space.Space describing
        the format of the vector space that the model outputs
        (this is a generalization of get_output_dim) """

        return self.output_space

    def free_energy(self, V):
        """
        Compute the free energy of data examples, if this model has
        probabilistic semantics.

        Parameters
        ----------
        V : tensor_like, 2-dimensional
            A batch of i.i.d. examples with examples indexed along the
            first axis and features along the second. This is data on which
            the monitoring quantities will be calculated (e.g., a validation
            set).

        Returns
        -------
        free_energy : tensor, 1-dimensional
            A (symbolic) vector of free energies for each data example in
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
            Flag to be passed to the `.get_value()` method of the
            shared variable. If `False`, a copy will always be returned.

        Returns
        -------
        params : list
            A list of `numpy.ndarray` objects containing the current
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
        return [param.get_value(borrow=borrow) for param in self.get_params()]

    def set_param_values(self, values, borrow=False):
        """
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
        """ Returns the number of visible units of the model.
        Deprecated; this assumes the model operates on a vector.
        Use get_input_space instead """
        raise NotImplementedError()

    def get_output_dim(self):
        """ Returns the number of visible units of the model.
        Deprecated; this assumes the model operates on a vector.
        Use get_input_space instead """
        raise NotImplementedError()

    def __getstate__(self):
        """
        This is the method that pickle/cPickle uses to determine what
        portion of the model to serialize. We remove all fields listed in
        `self.fields_to_del`. In particular, this should include all Theano
        functions, since they do not play nice with pickling.
        """
        d = {}
        names_to_del = getattr(self, 'names_to_del', set())
        names_to_keep = set(self.__dict__.keys()).difference(names_to_del)
        for name in names_to_keep:
            d[name] = self.__dict__[name]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __init__(self):
        self.names_to_del = set()
        self._test_batch_size = 2

    def get_test_batch_size(self):
        """ Batches of examples used to initialize
            X.tag.test_value should have this many
            examples if used as input to the model.
            (The model specifies the number of examples
            in case it needs a fixed batch size or to
            keep the memory usage of testing under control)"""
        return self._test_batch_size

    def register_names_to_del(self, names):
        """
        Register names of fields that should not be pickled.

        Parameters
        ----------
        names : iterable
            A collection of strings indicating names of fields on this
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

    def set_dtype(self, dtype, parent_name = ""):
        """
        Sets the dtype of any shared variables.

        Parameters
        ----------
        dtype : object or str
            A NumPy dtype object, or string representing a known dtype.
        """

        warnings.warn("""This method is not safe.
                To change the dtype of a shared variable it is necessary to
                allocate a new shared variable. When this method changes
                the type of a shared variable, other objects might keep
                pointing at the old shared variable. For example, in a
                DBM two different RBM objects might share the same shared
                variable to represent the bias term of one layer of the
                DBM. Calling set_dtype on the DBM would result in both
                RBMs having their own shared variable for that bias term.""")

        for field in dir(self):
            obj = getattr(self, field)
            if hasattr(obj, 'get_value'):
                setattr(self, field, shared(np.cast[dtype](obj.get_value()),name = obj.name))
            if hasattr(obj, 'set_dtype'):
                if obj.set_dtype.im_self is not None:
                    obj.set_dtype(dtype, parent_name + '.' + field)

            if isinstance(obj, tuple):
                raise NotImplementedError("tuples aren't mutable so we need to write code to replace the whole thing if any of its elements needs replacing")

            if isinstance(obj,list):
                for i, elem in enumerate(obj):
                    if hasattr(elem, 'get_value'):
                        obj[i] =  shared(np.cast[dtype](elem.get_value()), name = elem.name)
                    elif hasattr(elem, 'set_dtype'):
                        elem.set_dtype(dtype, parent_name + '.' + field + '[]')

        for param in self.get_params():
            if param.type.dtype != dtype:
                raise AssertionError(str(self)+' failed to set '+str(param)+\
                        'of type '+str(type(param))+' to '+str(dtype))
