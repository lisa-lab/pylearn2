from theano import tensor as T
import copy


class Model(object):
    def train(self, dataset):
        """
        Optional method.
        If implemented, performs one epoch of training.

        Parameters
        ----------
        dataset: The pylearn2.datasets.dataset.Dataset object to draw training
                data from

        Return value:
            True if the method should be called again for another epoch
            False if convergence has been reached
        """

    def get_monitoring_channels(self, V):
        """
            V: a batch of IID examples, of shape (# examples, #features)
                this is a theano matrix representing a batch from the
                monitoring dataset

            return value:
                dictionary mapping channel names to symbolic values to
                be computed from V

            You can make any channel names you want, just try to make
            sure they won't collide with names made by the training
            Cost, etc. Anything you think is worth monitoring during
            training can be added here. You probably want to control
            which channels get added with some config option for your
            model.
        """

        return {}

    def score(self, V):
        """
            V: a batch of IID examples, of shape (# examples, #features)

            If the model implements a probability distribution on
            R^n, this method should return the gradient of the log probability
            of the batch with respect to V, or raise an exception explaining
            why this is not possible.
        """

        return T.grad(- self.free_energy(V).sum(), V)

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

    def free_energy(self, V):
        """
        V: a batch of examples, of shape (# examples, #features)

        If the model implements a probability distribution on R^n,
        this method should return a vector such that rval[i] is
        the free energy on V[i,:] """

        raise NotImplementedError()

    def get_params(self):
        """
        Returns the parameters the define the model.

        This is the main  mechanism by which generic training
        algorithms like SGD know which values to update, however,
        even model parameters that should not be learned ought to
        be included here, so that the model's parameter set is
        more predictable.

        Parameters may be included here but held constant during
        learning via the censor_updates method.
        """
        raise NotImplementedError()

    def get_param_values(self, borrow=False):
        """
        Returns the values of the parameters that define the model
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
        Re-compiles all theano functions used internally by the model.
        This function is often called after a model is unpickled from
        file, since theano functions are not pickled. However, it is
        not always called. This allows scripts like show_weights.py
        to rapidly unpickle a model and inspect its weights without
        needing to recompile all of its learning machinery.

        All theano functions compiled by this method should be registered
        with the register_names_to_del method.
        """
        pass

    def get_input_dim(self):
        raise NotImplementedError()

    def __getstate__(self):
        """
        This is the method that cpickle uses to determine
        what portion of the model to serialize. We remove
        all fields listed in self.fields_to_del.
        In particular, this should include all theano functions,
        since they do not play nice with pickling. """

        d = {}

        names_to_keep = set(self.__dict__.keys()).difference(self.names_to_del)

        for name in names_to_keep:
            d[name] = copy.copy(self.__dict__[name])

        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __init__(self):
        self.names_to_del = set()

    def register_names_to_del(self, names):
        """
        names: a list of names of fields that should not be pickled

        all names registered will be deleted from the dictionary
        returned by the model's __getstate__ method
        (unless your model overrides __getstate__ )
        """

        self.names_to_del = self.names_to_del.union(names)
