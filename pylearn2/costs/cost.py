""" Classes representing loss functions.
Currently, these are primarily used to specify
the objective function for the SGD algorithm."""
import theano.tensor as T


class Cost(object):
    """
    Represents a cost that can be called either as a supervised cost or an
    unsupervised cost.
    """
    def __init__(self):
        raise NotImplementedError("You should implement a constructor " + \
                                  "which at least sets a boolean value " + \
                                  "for the 'self.supervised' attribute.")

    def __call__(self, model, X, Y=None):
        raise NotImplementedError()

    def get_monitoring_channels(self, model, X, Y=None):
        return {}

    def get_target_space(self, model, dataset):
        if self.supervised:
            return model.get_output_space()
        else:
            return None


class SumOfCosts(Cost):
    """
    Combines multiple costs by summing them.
    """
    def __init__(self, costs):
        """
        Initialize the SumOfCosts object and make sure that the list of costs
        contains only Cost instances.

        Parameters
        ----------
        costs: list
            List of Cost objects
        """
        self.supervised = False
        self.costs = costs
        # Check whether the sum is a supervised cost and if all the costs are
        # Cost instances
        for cost in self.costs:
            if isinstance(cost, Cost):
                if cost.supervised:
                    self.supervised = True
            else:
                raise ValueError("one of the costs is not " + \
                                 "Cost instance")

    def __call__(self, model, X, Y=None):
        """
        Returns the sum of the costs the SumOfCosts instance was given at
        initialization.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        X : tensor_like
            input to the model
        Y : tensor_like
            the target, if necessary
        """
        # If the sum is a supervised cost, check whether the target was
        # provided
        if Y is None and self.supervised is True:
            raise ValueError("no targets provided while some of the " + \
                             "costs in the sum are supervised costs")
        sum_of_costs = 0
        for cost in self.costs:
            if cost.supervised:
                sum_of_costs = sum_of_costs + cost(model, X, Y)
            else:
                sum_of_costs = sum_of_costs + cost(model, X)
        return sum_of_costs


class ScaledCost(Cost):
    """
    Represents a given cost scaled by a constant factor.
    """
    def __init__(self, cost, scaling):
        """
        Parameters
        ----------
        cost: Cost
            cost to be scaled
        scaling : float
            scaling of the cost
        """
        self.cost = cost
        self.supervised = cost.supervised
        self.scaling = scaling

    def __call__(self, model, X, Y=None):
        """
        Returns cost scaled by its scaling factor.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the scaled cost
        X : tensor_like
            input to the model
        Y : tensor_like
            the target, if necessary
        """
        if Y is None and self.supervised is True:
            raise ValueError("no targets provided for a supervised cost")
        if self.supervised:
            return self.scaling * self.cost(model, X, Y)
        else:
            return self.scaling * self.cost(model, X)


class LxReg(Cost):
    """
    L-x regularization term for the list of tensor variables provided.
    """
    def __init__(self, variables, x):
        """
        Initialize LxReg with the variables and scaling provided.

        Parameters:
        -----------
        variables: list
            list of tensor variables to be regularized
        x: int
            the x in "L-x regularization""
        """
        self.supervised = False
        self.variables = variables
        self.x = x

    def __call__(self, model=None, X=None, Y=None):
        """
        Return the scaled L-x regularization term. The optional parameters are
        never used, they're there only to provide an interface consistent with
        both SupervisedCost and UnsupervisedCost.
        """
        Lx = 0
        for var in self.variables:
            Lx = Lx + abs(var ** self.x).sum()
        return Lx


class CrossEntropy(Cost):
    """WRITEME"""
    def __init__(self):
        self.supervised = True

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
                "GeneralCost instance (%s of type %s)."
                " This probably means the superclass you provided isn't a "
                "subclass of Cost.") % (str(rval),str(type(rval))))
        # TODO: is there a way to directly check superclass?
    return rval
