"""
Classes representing loss functions.
Currently, these are primarily used to specify
the objective function for the SGD and BGD
training algorithms.
"""
import theano.tensor as T
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

class Cost(object):
    """
    Represents a cost that can be called either as a supervised cost or an
    unsupervised cost.
    """

    # If True, the Y argument to __call__ and get_gradients must not be None
    supervised = False

    def __call__(self, model, X, Y=None, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        Returns a symbolic expression for a cost function applied to the
        minibatch of data.
        Optionally, may return None. This represents that the cost function
        is intractable but may be optimized via the get_gradients method.

        """

        raise NotImplementedError(str(type(self))+" does not implement __call__")

    def get_gradients(self, model, X, Y=None, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """

        try:
            if Y is None:
                cost = self(model=model, X=X, **kwargs)
            else:
                cost = self(model=model, X=X, Y=Y, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates

    def get_monitoring_channels(self, model, X, Y=None, **kwargs):
        """
        Returns a dictionary mapping channel names to expressions for
        channel values.

        WRITEME: how do you do prereqs in this setup? (there is a way,
            but I forget how right now)

        model: the model to use to compute the monitoring channels
        X, Y: symbolic expressions for the monitoring data
        kwargs: used so that custom algorithms can use extra variables
                for monitoring.

        """
        return OrderedDict()

    def get_target_space(self, model, dataset):
        if self.supervised:
            return model.get_output_space()
        else:
            return None

    def get_fixed_var_descr(self, model, X, Y):
        """
        Subclasses should override this if they need variables held
        constant across multiple updates to a minibatch.

        TrainingAlgorithms that do multiple updates to a minibatch should
        respect this. See FixedVarDescr below for details.
        """

        return FixedVarDescr()


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
            List of Cost objects or (coeff, Cost) pairs
        """
        assert isinstance(costs, list)
        assert len(costs) > 0

        self.costs = []
        self.coeffs = []

        for cost in costs:
            if isinstance(cost, (list, tuple)):
                coeff, cost = cost
            else:
                coeff = 1.
            self.coeffs.append(coeff)
            self.costs.append(cost)
            if not isinstance(cost, Cost):
                raise ValueError("one of the costs is not " + \
                                 "Cost instance")

        self.supervised = any([cost.supervised for cost in self.costs])

    def __call__(self, model, X, Y=None, ** kwargs):
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
            raise ValueError("no targets provided while some of the " +
                             "costs in the sum are supervised costs")

        costs = []
        for cost in self.costs:
            if cost.supervised:
                Y_to_pass = Y
            else:
                Y_to_pass = None
            costs.append(cost(model, X, Y_to_pass, **kwargs))
        assert len(costs) > 0

        if any([cost is None for cost in costs]):
            sum_of_costs = None
        else:
            costs = [coeff * cost for coeff, cost in safe_zip(self.coeffs, costs)]
            assert len(costs) > 0
            sum_of_costs = reduce(lambda x, y: x + y, costs)

        return sum_of_costs

    def get_gradients(self, model, X, Y=None, ** kwargs):

        if Y is None and self.supervised:
            raise ValueError("no targets provided while some of the " +
                             "costs in the sum are supervised costs")

        indiv_results = []
        for cost in self.costs:
            if cost.supervised:
                Y_to_pass = Y
            else:
                Y_to_pass = None
            result = cost.get_gradients(model, X, Y_to_pass, ** kwargs)
            indiv_results.append(result)


        grads = OrderedDict()
        updates = OrderedDict()

        params = model.get_params()

        for coeff, packed in zip(self.coeffs, indiv_results):
            g, u = packed
            for param in g:
                if param not in params:
                    raise ValueError("A shared variable ("+str(param)+") that is not a parameter appeared in a cost gradient dictionary.")
            for param in g:
                assert param.ndim == g[param].ndim
                v = coeff * g[param]
                if param not in grads:
                    grads[param] = v
                else:
                    grads[param] = grads[param] + v
                assert grads[param].ndim == param.ndim
            assert not any([state in updates for state in u])
            assert not any([state in params for state in u])
            updates.update(u)

        return grads, updates

    def get_monitoring_channels(self, model, X, Y=None, ** kwargs):
        if Y is  None and self.supervised:
            raise ValueError("no targets provided while some of the " +
                             "costs in the sum are supervised costs")

        rval = OrderedDict()

        for i, cost in enumerate(self.costs):
            try:
                rval.update(cost.get_monitoring_channels(model, X, Y, **kwargs))
            except TypeError:
                print 'SumOfCosts.get_monitoring_channels encountered TypeError while calling ' \
                        + str(type(cost))+'.get_monitoring_channels'
                raise

            Y_to_pass = Y
            if not cost.supervised:
                Y_to_pass = None

            value = cost(model, X, Y_to_pass, ** kwargs)
            if value is not None:
                name = ''
                if hasattr(value, 'name') and value.name is not None:
                    name = '_' + value.name
                rval['term_'+str(i)+name] = value

        return rval

    def get_fixed_var_descr(self, model, X, Y):

        assert Y is not None

        descrs = [cost.get_fixed_var_descr(model, X, Y) for cost in self.costs]

        return reduce(merge, descrs)



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

    def __call__(self, model, X, Y, ** kwargs):
        """WRITEME"""
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()


class MethodCost(Cost):
    """
    A cost specified via the string name of a method of the model.
    """

    def __init__(self, method, supervised = False):
        """
            method: a string specifying the name of the method of the model
                    that should be called to generate the objective function.
        """
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, *args, **kwargs):
            """ Patches calls through to a user-specified method of the model """
            fn = getattr(model, self.method)
            return fn(*args, **kwargs)


def _no_op(X, y=None):
    """
    An on_load_batch callback that does nothing.
    """

class FixedVarDescr(object):
    """
    An object used to describe variables that influence the cost but that should
    be held fixed for each minibatch, even if the learning algorithm makes multiple
    changes to the parameters on this minibatch, ie, for a line search, etc.
    """

    def __init__(self):
        """
        fixed_vars: maps string names to shared variables or some sort of data structure
                    surrounding shared variables.
                    Any learning algorithm that does multiple updates on the same minibatch
                    should pass fixed_vars to the cost's __call__ and get_gradient methods
                    as keyword arguments.
        """
        self.fixed_vars = {}

        """
        A list of callable objects that the learning algorithm should
        call with X or X and y as appropriate
        whenever a new batch of data is loaded.
        This will update the shared variables mapped to by fixed_vars.
        """
        self.on_load_batch = [_no_op]

def merge(left, right):
    """
    Combine two FixedVarDescrs
    """

    assert left is not right
    # We assume aliasing is a bug
    assert left.fixed_vars is not right.fixed_vars
    assert left.on_load_batch is not right.on_load_batch

    rval = FixedVarDescr()
    for key in left.fixed_vars:
        if key in right.fixed_vars:
            raise ValueError("Can't merge these FixedVarDescrs, both contain "+key)
    assert not any([key in left.fixed_vars for key in right.fixed_vars])
    rval.fixed_vars.update(left.fixed_vars)
    rval.fixed_vars.update(right.fixed_vars)

    rval.on_load_batch = safe_union(left.on_load_batch, right.on_load_batch)

    return rval


    def __call__(self, X, Y):
        return self.wrapped(X)
