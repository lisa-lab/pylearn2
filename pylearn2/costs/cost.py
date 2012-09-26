""" Classes representing loss functions.
Currently, these are primarily used to specify
the objective function for the SGD algorithm."""
import theano.tensor as T


class Cost(object):
    """
    Represents a cost function.
    """
    def __init__(self):
        self.supervised = None

class SupervisedCost(Cost):
    """
    Represents a supervised cost, i.e. a cost which uses both the model's
    output and the target in its computation.
    """
    def __init__(self):
        self.supervised = True

    def get_monitoring_channels(self, model, X, Y):
        return {}

    def get_target_space(self, model, dataset):
        return model.get_output_space()

    def __call__(self, model, X, Y):
        """
            model: the model the cost is applied to
            X: a batch of features. First axis indexes examples
            Y: a batch of labels corresponding to X

            Returns a symbolic expression for the loss function.
        """
        raise NotImplementedError(str(self)+" does not implement __call__")


class UnsupervisedCost(Cost):
    """Abstract class representing a cost of features only"""

    def __init__(self):
        self.supervised = False

    def get_monitoring_channels(self, model, X):
        return {}

        def get_target_space(self, model, datatset):
            return None

    def __call__(self, model, X):
        """
            model: the model the cost is applied to
            X: a batch of features. First axis indexes examples

            Returns a symbolic expression for the loss function.
        """
        raise NotImplementedError(str(self)+" does not implement __call__")

class CrossEntropy(SupervisedCost):
    """WRITEME"""
    def __call__(self, model, X, Y):
        """WRITEME"""
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()


def make_method_cost(method, superclass):
    """
        A cost specified via the string name of a method of the model.
        Makes a new class derived from superclass.
        Assumes all it needs to implement is __call__ and that the
        first argument to __call__ is called 'model'.
        Implements this method by passing the remaining arguments
        to getattr(model, method)

        Example usage:

        class MyCrazyNewModel(Model):

            def my_crazy_new_loss_function(self, X):

                ...

        my_cost = method_cost('my_crazy_new_loss_function', UnsupervisedCost)
    """
    class MethodCost(superclass):
        """ A Cost defined by the name of a model's method """
        def __call__(self, model, *args, **kwargs):
            """ Patches calls through to a user-specified method of the model """
            fn = getattr(model, method)
            return fn(*args, **kwargs)

    rval = MethodCost()
    if not isinstance(rval, Cost):
        raise TypeError(("make_method_cost made something that isn't a "
                "Cost instance (%s of type %s)."
                " This probably means the superclass you provided isn't a "
                "subclass of Cost.") % (str(rval),str(type(rval))))
        # TODO: is there a way to directly check superclass?
    return rval


class GeneralCost(Cost):
    """
    Represents a cost that can be called either as a supervised cost or an
    unsupervised cost.
    """
    def __call__(self, model, X, Y=None):
        raise NotImplementedError()

    def get_monitoring_channels(self, model, X, Y=None):
        return {}
