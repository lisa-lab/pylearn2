"""
Classes representing cost functions.

Currently, these are primarily used to specify the objective function for
the SGD and BGD training algorithms.
"""

import functools
import logging
import warnings
from itertools import izip

import theano.tensor as T
from theano.compat.python2x import OrderedDict

from pylearn2.utils import safe_zip
from pylearn2.utils import safe_union
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.utils.data_specs import DataSpecsMapping


logger = logging.getLogger(__name__)

class Cost(object):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.

    Notes
    -----
    While functions may be represented just as theano graphs,
    this class allows us to add extra functionality. The
    `get_gradients` methods allows us to use a method other
    than `theano.tensor.grad` to compute the gradient of the
    cost function. This enables using approximate gradients
    of cost functions that are not differentiable or whose
    true gradient is computationally intractable. Additionally,
    the get_monitoring_channels method allows monitoring of
    quantities that are useful when training with a given
    objective function (such as termination criteria that are
    usually used with the function).
    """

    # TODO: remove this when it is no longer necessary, it should be
    # mostly phased out due to the new data_specs interface.
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = False

    def expr(self, model, data, ** kwargs):
        """
        Returns a theano expression for the cost function.

        Returns a symbolic expression for a cost function applied to the
        minibatch of data.
        Optionally, may return None. This represents that the cost function
        is intractable but may be optimized via the get_gradients method.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments. Not used by the base class.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "expr.")

    def get_gradients(self, model, data, ** kwargs):
        """
        Provides the gradients of the cost function with respect to the model
        parameters.

        These are not necessarily those obtained by theano.tensor.grad
        --you may wish to use approximate or even intentionally incorrect
        gradients in some cases.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments, not used by the base class.

        Returns
        -------
        gradients : OrderedDict
            a dictionary mapping from the model's parameters
            to their gradients
            The default implementation is to compute the gradients
            using T.grad applied to the value returned by expr.
            However, subclasses may return other values for the gradient.
            For example, an intractable cost may return a sampling-based
            approximation to its gradient.
        updates : OrderedDict
            a dictionary mapping shared variables to updates that must
            be applied to them each time these gradients are computed.
            This is to facilitate computation of sampling-based approximate
            gradients.
            The parameters should never appear in the updates dictionary.
            This would imply that computing their gradient changes
            their value, thus making the gradient value outdated.
        """

        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError, e:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling " + str(type(self)) + ".expr"
            logger.error(type(self))
            logger.error(e.message)
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates

    def get_monitoring_channels(self, model, data, **kwargs):
        """
        .. todo::

            WRITEME

        .. todo::

            how do you do prereqs in this setup? (I think PL changed
            it, not sure if there still is a way in this context)

        Returns a dictionary mapping channel names to expressions for
        channel values.

        Parameters
        ----------
        model : Model
            the model to use to compute the monitoring channels
        data : batch
            (a member of self.get_data_specs()[0])
            symbolic expressions for the monitoring data
        kwargs : dict
            used so that custom algorithms can use extra variables
            for monitoring.

        Returns
        -------
        rval : dict
            Maps channels names to expressions for channel values.
        """
        self.get_data_specs(model)[0].validate(data)
        return OrderedDict()

    def get_fixed_var_descr(self, model, data):
        """
        Subclasses should override this if they need variables held
        constant across multiple updates to a minibatch.

        TrainingAlgorithms that do multiple updates to a minibatch should
        respect this. See the FixedVarDescr class for details.

        Parameters
        ----------
        model : Model
        data : theano.gof.Variable or tuple
            A valid member of the Space used to train `model` with this
            cost.

        Returns
        -------
        fixed_var_descr : FixedVarDescr
            A description of how to hold the necessary variables constant
        """
        self.get_data_specs(model)[0].validate(data)
        fixed_var_descr = FixedVarDescr()
        return fixed_var_descr

    def get_data_specs(self, model):
        """
        Returns a specification of the Space the data should lie in and
        its source (what part of the dataset it should come from).

        Parameters
        ----------
        model : Model
            The model to train with this cost

        Returns
        -------
        data_specs : tuple
            The tuple should be of length two.
            The first element of the tuple should be a Space (possibly a
            CompositeSpace) describing how to format the data.
            The second element of the tuple describes the source of the
            data. It probably should be a string or nested tuple of strings.

        See Also
        --------
        For many common cases, rather than implementing this method
        yourself, you probably want
        to just inherit from `DefaultDataSpecsMixin` or NullDataSpecsMixin.

        Notes
        -----
        .. todo

            figure out return format for sure. PL seems to have documented
            this method incorrectly.

        """
        raise NotImplementedError(str(type(self)) + " does not implement " +
                                  "get_data_specs.")

class SumOfCosts(Cost):
    """
    Combines multiple costs by summing them.

    Parameters
    ----------
    costs : list
        List of Cost objects or (coeff, Cost) pairs
    """

    def __init__(self, costs):
        """
        Initialize the SumOfCosts object and make sure that the list of costs
        contains only Cost instances.

        Parameters
        ----------
        costs : list
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
                raise ValueError("one of the costs is not "
                                 "Cost instance")

        # TODO: remove this when it is no longer necessary
        self.supervised = any([cost_.supervised for cost_ in self.costs])

    def expr(self, model, data, ** kwargs):
        """
        Returns the sum of the costs the SumOfCosts instance was given at
        initialization.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        data : flat tuple of tensor_like variables.
            data has to follow the format defined by self.get_data_specs(),
            but this format will always be a flat tuple.
        """
        self.get_data_specs(model)[0].validate(data)
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)
        costs = []
        for cost, cost_data in safe_zip(self.costs, nested_data):
            costs.append(cost.expr(model, cost_data, **kwargs))
        assert len(costs) > 0

        if any([cost is None for cost in costs]):
            sum_of_costs = None
        else:
            costs = [coeff * cost
                     for coeff, cost in safe_zip(self.coeffs, costs)]
            assert len(costs) > 0
            sum_of_costs = reduce(lambda x, y: x + y, costs)

        return sum_of_costs

    def get_composite_data_specs(self, model):
        """
        Build and return a composite data_specs of all costs.

        The returned space is a CompositeSpace, where the components are
        the spaces of each of self.costs, in the same order. The returned
        source is a tuple of the corresponding sources.

        Parameters
        ----------
        model : pylearn2.models.Model
        """
        spaces = []
        sources = []
        for cost in self.costs:
            space, source = cost.get_data_specs(model)
            spaces.append(space)
            sources.append(source)

        # Build composite space representing all inputs
        composite_space = CompositeSpace(spaces)
        sources = tuple(sources)
        return (composite_space, sources)

    def get_composite_specs_and_mapping(self, model):
        """
        Build the composite data_specs and a mapping to flatten it, return both

        Build the composite data_specs described in `get_composite_specs`, and
        build a DataSpecsMapping that can convert between it and a flat
        equivalent version. In particular, it helps building a flat data_specs
        to request data, and nesting this data back to the composite
        data_specs, so it can be dispatched among the different sub-costs.

        Parameters
        ----------
        model : pylearn2.models.Model

        Notes
        -----
        This is a helper function used by `get_data_specs` and `get_gradients`,
        and possibly other methods.
        """
        composite_space, sources = self.get_composite_data_specs(model)
        mapping = DataSpecsMapping((composite_space, sources))
        return (composite_space, sources), mapping

    def get_data_specs(self, model):
        """
        Get a flat data_specs containing all information for all sub-costs.

        Parameters
        ----------
        model : pylearn2.models.Model
            TODO WRITEME

        Notes
        -----
        This data_specs should be non-redundant. It is built by flattening
        the composite data_specs returned by `get_composite_specs`.

        This is the format that SumOfCosts will request its data in. Then,
        this flat data tuple will be nested into the composite data_specs,
        in order to dispatch it among the different sub-costs.
        """
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        composite_space, sources = composite_specs
        flat_composite_space = mapping.flatten(composite_space)
        flat_sources = mapping.flatten(sources)
        data_specs = (flat_composite_space, flat_sources)
        return data_specs

    @functools.wraps(Cost.get_gradients)
    def get_gradients(self, model, data, ** kwargs):
        indiv_results = []
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)
        for cost, cost_data in safe_zip(self.costs, nested_data):
            result = cost.get_gradients(model, cost_data, ** kwargs)
            indiv_results.append(result)

        grads = OrderedDict()
        updates = OrderedDict()
        params = model.get_params()

        for coeff, packed in zip(self.coeffs, indiv_results):
            g, u = packed
            for param in g:
                if param not in params:
                    raise ValueError("A shared variable (" +
                                     str(param) +
                                     ") that is not a parameter appeared "
                                     "a cost gradient dictionary.")
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

    @functools.wraps(Cost.get_monitoring_channels)
    def get_monitoring_channels(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)

        for i, cost in enumerate(self.costs):
            cost_data = nested_data[i]
            try:
                channels = cost.get_monitoring_channels(model, cost_data,
                                                        **kwargs)
                rval.update(channels)
            except TypeError:
                logger.error('SumOfCosts.get_monitoring_channels encountered '
                             'TypeError while calling {0}'
                             '.get_monitoring_channels'.format(type(cost)))
                raise

            value = cost.expr(model, cost_data, ** kwargs)
            if value is not None:
                name = ''
                if hasattr(value, 'name') and value.name is not None:
                    name = '_' + value.name
                rval['term_' + str(i) + name] = value

        return rval

    def get_fixed_var_descr(self, model, data):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        model : Model
        data : theano.gof.Variable or tuple
            A valid member of the Space defined by
            self.get_data_specs(model)[0]
        """
        data_specs = self.get_data_specs(model)
        data_specs[0].validate(data)
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)

        descrs = [cost.get_fixed_var_descr(model, cost_data)
                  for cost, cost_data in safe_zip(self.costs, nested_data)]

        return reduce(merge, descrs)

def scaled_cost(cost, scaling):
    """
    Deprecated. Switch to SumOfCosts([[scaling, cost]]), or just quit using it.

    Parameters
    ----------
    cost : Cost
        cost to be scaled
    scaling : float
        scaling of the cost
    """

    warnings.warn("""\
scaled_cost is deprecated and may be removed on or after 2014-08-05.
SumOfCosts allows you to scale individual terms, and if this is the only cost,
you may as well just change the learning rate.""")

    return SumOfCosts([[scaling, cost]])

class NullDataSpecsMixin(object):
    """
    Use multiple inheritance with both this object and Cost in order to
    obtain a data specification corresponding to not using data at all.

    Due to method resolution order, you want Cost to appear after
    NullDataSpecsMixin in the superclass list.
    """

    def get_data_specs(self, model):
        """
        Provides an implementation of `Cost.expr`.

        Returns data specifications corresponding to not using any
        data at all.

        Parameters
        ----------
        model : pylearn2.models.Model
        """
        return (NullSpace(), '')

class DefaultDataSpecsMixin(object):
    """
    Use multiple inheritance with both this object and Cost in order to
    obtain the default data specification.

    Due to method resolution order, you want Cost to appear after
    DefaultDataSpecsMixin in the superclass list.
    """

    def get_data_specs(self, model):
        """
        Provides a default data specification.

        The cost requests input features from the model's input space and
        input source. `self` must contain a bool field called `supervised`.
        If this field is True, the cost requests targets as well.

        Parameters
        ----------
        model : pylearn2.models.Model
            TODO WRITEME
        """
        if self.supervised:
            space = CompositeSpace([model.get_input_space(),
                                    model.get_output_space()])
            sources = (model.get_input_source(), model.get_target_source())
            return (space, sources)
        else:
            return (model.get_input_space(), model.get_input_source())

class LpPenalty(NullDataSpecsMixin, Cost):
    """
    L-p penalty of the tensor variables provided.

    Parameters
    ----------
    variables : list
        list of tensor variables to be regularized
    p : int
        p in "L-p penalty"
    """
    def __init__(self, variables, p):
        """
        Parameters
        ----------
        variables : list
            list of tensor variables to be regularized
        p : int
            p in "L-p penalty"
        """
        self.variables = variables
        self.p = p

    def expr(self, model, data, **kwargs):
        """
        Return the L-p penalty term. The optional parameters are never used;
        they're only there to provide an interface that's consistent with
        the Cost superclass.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments. Not used by the base class.
        """
        # This Cost does not depend on any data, and get_data_specs does not
        # ask for any data, so we should not be provided with some.
        self.get_data_specs(model)[0].validate(data)

        penalty = 0
        for var in self.variables:
            # Absolute value handles odd-valued p cases
            penalty = penalty + abs(var ** self.p).sum()
        return penalty


class CrossEntropy(DefaultDataSpecsMixin, Cost):
    """
    DEPRECATED
    """

    def __init__(self):
        warnings.warn("CrossEntropy is deprecated. You should use a "
                "model-specific cross entropy cost function. CrossEntropy"
                "will be removed on or after August 3, 2014", stacklevel=2)
        self.supervised = True

    def expr(self, model, data, ** kwargs):
        """
        DEPRECATED

        Parameters
        ----------
        model : DEPRECATED
        data : DEPRECATED
        """
        self.get_data_specs(model)[0].validate(data)

        # unpack data
        (X, Y) = data
        return (-Y * T.log(model(X)) -
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()


class MethodCost(Cost):
    """
    A cost specified via the string name of a method of the model.

    Parameters
    ----------
    method : a string specifying the name of the method of the model
            that should be called to generate the objective function.
    data_specs : a string specifying the name of a method/property of
            the model that describe the data specs required by
            method
    """

    def __init__(self, method, data_specs=None):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        method : a string specifying the name of the method of the model
                that should be called to generate the objective function.
        data_specs : a string specifying the name of a method/property of
                the model that describe the data specs required by
                method
        """
        self.method = method
        self.data_specs = data_specs

    def expr(self, model, data, *args, **kwargs):
        """
        Patches calls through to a user-specified method of the model

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        data : flat tuple of tensor_like variables.
            data has to follow the format defined by self.get_data_specs(),
            but this format will always be a flat tuple.
        """
        self.get_data_specs(model)[0].validate(data)
        fn = getattr(model, self.method)
        return fn(data, *args, **kwargs)

    @functools.wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        if self.data_specs is not None:
            fn = getattr(model, self.data_specs)
        else:
            # To be compatible with earlier scripts,
            # try (self.method)_data_specs
            fn = getattr(model, '%s_data_specs' % self.method)

        if callable(fn):
            return fn()
        else:
            return fn


def _no_op(data):
    """
    An on_load_batch callback that does nothing.
    """

class FixedVarDescrDataSpecsError(TypeError):
    """
    An error raised when code attempts to use the unused
    data_specs field of FixedVarDescr
    """

class FixedVarDescr(object):
    """
    An object used to describe variables that influence the cost but that
    should be held fixed for each minibatch, even if the learning algorithm
    makes multiple changes to the parameters on this minibatch, i.e., for a
    line search, etc.

    Attributes

    - fixed_vars : dict
        maps string names to shared variables or some sort of data
        structure surrounding shared variables.
        Any learning algorithm that does multiple updates on the same
        minibatch should pass fixed_vars to the cost's expr and
        get_gradient methods as keyword arguments.
    - on_load_batch : list
        A list of callable objects that the learning algorithm should
        call with input data.
        All of these callables must take an argument with the same
        (space, source) format as the cost used for training.
        TODO: It can be hard for a human user to know the right format
        ahead of time if you use SumOfCosts, make a better way of handling
        this.
        PL had added a data_specs field to this class which
        was meant to define the (space, source) format for each of
        the members of on_load_batch, but the doc was internally
        inconsistent, none of the TrainingAlgorithms obeyed it,
        and the Cost's handling of it was buggy. IG removed this
        broken functionality so that at least singleton costs can
        used FixedVarDescr but it would be good to restore functionality
        to composite costs.
    """

    def __init__(self):
        self.fixed_vars = {}
        self.on_load_batch = []

    def _data_specs_err(self, x = None):
        raise FixedVarDescrDataSpecsError("The data_specs field of "
                "FixedVarDescr has been "
                "removed. While this field existed and was documented at "
                "one time, no TrainingAlgorithm respected it. The "
                "data_specs of all members of on_load_batch must match "
                "those of the cost.")

    data_specs = property(_data_specs_err, _data_specs_err)


def merge(left, right):
    """
    Combine two FixedVarDescrs

    Parameters
    ----------
    left : FixedVarDescr
    right : FixedVarDescr

    Returns
    -------
    merged : FixedVarDescr
        a new FixedVarDescr describing all variables and operations
        described by `left` and `right`
    """

    # We assume aliasing is a bug
    assert left is not right
    assert left.fixed_vars is not right.fixed_vars
    assert left.on_load_batch is not right.on_load_batch

    merged = FixedVarDescr()
    for key in left.fixed_vars:
        if key in right.fixed_vars:
            raise ValueError("Can't merge these FixedVarDescrs, "
                             "both contain " + key)
    assert not any([key in left.fixed_vars for key in right.fixed_vars])
    merged.fixed_vars.update(left.fixed_vars)
    merged.fixed_vars.update(right.fixed_vars)

    merged.on_load_batch = safe_union(left.on_load_batch,
                                        right.on_load_batch)

    return merged
