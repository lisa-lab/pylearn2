"""
Code to hook into the MLP framework
"""
import functools
import inspect
import logging

from pylearn2.sandbox.rnn.space import SequenceSpace, SequenceDataSpace
from pylearn2.space import CompositeSpace
from pylearn2.utils.track_version import MetaLibVersion

log = logging.getLogger(__name__)

# These layers are guaranteed to be wrapped without issues
WHITELIST = [
    'Softmax',
    'SoftmaxPool',
    'Linear',
    'ProjectionLayer',
    'Tanh',
    'Sigmoid',
    'RectifiedLinear',
    'Softplus',
    'SpaceConverter',
    'WindowLayer'
]

# These can't be wrapped
BLACKLIST = [
    'CompositeLayer',
    'FlattenerLayer'  # Double check this
]


class RNNWrapper(MetaLibVersion):
    """
    This metaclass wraps the Layer class and all its children
    by intercepting the class creation. Methods can be wrapped by
    defining a `_wrapper` method.

    Note that the MLP class isn't wrapped in general, it suffices to
    wrap the component layers.

    Parameters
    ----------
    See https://docs.python.org/2/reference/datamodel.html#object.__new__
    """
    def __new__(cls, name, bases, dct):
        wrappers = [attr[:-8] for attr in cls.__dict__.keys()
                    if attr.endswith('_wrapper')]
        for wrapper in wrappers:
            if wrapper not in dct:
                for base in bases:
                    method = getattr(base, wrapper, None)
                    if method is not None:
                        break
            else:
                method = dct[wrapper]
            dct[wrapper] = getattr(cls, wrapper + '_wrapper')(name, method)

        dct['rnn_friendly'] = False
        dct['_requires_reshape'] = False
        dct['_requires_unmask'] = False
        return type.__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        """
        This method overrides MLP's __init__ method. It recursively
        goes through the spaces and sources, and for each SequenceSpace
        it adds an extra source for the mask. This is just syntactic sugar,
        preventing people from adding a source for each mask.

        Eventually this behaviour could maybe be moved to the
        DataSpecsMapping class. Alternatively, a subclass of MLP
        could be written that performs this action before calling
        its base's __init__ method.

        Parameters
        ----------
        See https://docs.python.org/2/reference/datamodel.html#object.__new__
        """
        super(RNNWrapper, cls).__init__(name, bases, dct)
        if name == 'MLP':
            old_init = dct['__init__']

            @functools.wraps(old_init)
            def __init__(self, layers, batch_size=None, input_space=None,
                         input_source='features', target_source='targets',
                         nvis=None, seed=None,
                         layer_name=None, **kwargs):
                def add_mask_source(input_space, input_source):
                    """
                    This is a recursive helper function to go
                    through the nested spaces and tuples
                    """
                    if isinstance(input_space, CompositeSpace):
                        if not isinstance(input_space, SequenceSpace):
                            input_source = tuple(
                                add_mask_source(component, source)
                                for component, source
                                in zip(input_space.components, input_source)
                            )
                        else:
                            assert isinstance(input_source, basestring)
                            input_source = (input_source,
                                            input_source + '_mask')
                    return input_source
                input_source = add_mask_source(input_space, input_source)
                return old_init(self, layers, batch_size, input_space,
                                input_source, target_source, nvis, seed, layer_name, **kwargs)
            cls.__init__ = __init__

    @classmethod
    def fprop_wrapper(cls, name, fprop):
        """
        If a layer receives a SequenceSpace it should receive
        a tuple of (data, mask). For layers that cannot deal with this
        we do the following:

        - Unpack (data, mask) and perform the fprop with the data only
        - Add the mask back just before returning, so that the next layer
          receives a tuple again

        Besides the mask, we also need to take are of reshaping the data.
        This reshaping needs to happen even if we receive SequenceDataSpace
        data instead of SequenceSpace data. The format is
        (time, batch, data, ..., data) which needs to be reshaped to
        (time * batch, data, ..., data) before calling the original fprop,
        after which we need to reshape it back.

        Parameters
        ----------
        fprop : method
            The fprop method to be wrapped
        """
        @functools.wraps(fprop)
        def outer(self, state_below):
            if self._requires_reshape:
                if self._requires_unmask:
                    state_below, mask = state_below
                input_shape = ([state_below.shape[0] *
                                state_below.shape[1]] +
                               [state_below.shape[i]
                                for i in xrange(2, state_below.ndim)])
                reshaped_state_below = state_below.reshape(input_shape)
                reshaped_state = fprop(self, reshaped_state_below)
                output_shape = ([state_below.shape[0],
                                 state_below.shape[1]] +
                                [reshaped_state.shape[i]
                                 for i in xrange(1, reshaped_state.ndim)])
                state = reshaped_state.reshape(output_shape)
                if self._requires_unmask:
                    return (state, mask)
                else:
                    return state
            else:  # Not RNN-friendly, but not requiring reshape
                return fprop(self, state_below)
        return outer

    @classmethod
    def cost_wrapper(cls, name, cost):
        """
        This layer wraps cost methods by reshaping the tensor (merging
        the time and batch axis) and then taking out all the masked
        values before applying the cost method.
        """
        @functools.wraps(cost)
        def outer(self, Y, Y_hat):
            if self._requires_reshape:
                if self._requires_unmask:
                    Y, Y_mask = Y
                    Y_hat, Y_hat_mask = Y_hat
                input_shape = ([Y.shape[0] * Y.shape[1]] +
                               [Y.shape[i] for i in xrange(2, Y.ndim)])
                reshaped_Y = Y.reshape(input_shape)
                reshaped_Y_hat = Y_hat.reshape(input_shape)
                # Here we need to take the indices of only the unmasked data
                if self._requires_unmask:
                    return cost(self, reshaped_Y[Y_mask.flatten().nonzero()],
                                reshaped_Y_hat[Y_mask.flatten().nonzero()])
                return cost(self, reshaped_Y, reshaped_Y_hat)
            else:  # Not RNN-friendly, but not requiring reshape
                return cost(self, Y, Y_hat)
        return outer

    @classmethod
    def cost_matrix_wrapper(cls, name, cost_matrix):
        """
        If the cost_matrix is called from within a cost function,
        everything is fine, since things were reshaped and unpacked.
        In any other case we raise a warning (after which it most likely
        crashes).
        """
        @functools.wraps(cost_matrix)
        def outer(self, Y, Y_hat):
            if self._requires_reshape and inspect.stack()[1][3] != 'cost':
                log.warning("You are using the `cost_matrix` method on a "
                            "layer which has been wrapped to accept sequence "
                            "input, might or might not be problematic.")
            return cost_matrix(self, Y, Y_hat)
        return outer

    @classmethod
    def cost_from_cost_matrix_wrapper(cls, name, cost_from_cost_matrix):
        """
        If the cost_from_cost_matrix is called from within a cost function,
        everything is fine, since things were reshaped and unpacked.
        In any other case we raise a warning (after which it most likely
        crashes).
        """
        @functools.wraps(cost_from_cost_matrix)
        def outer(self, cost_matrix):
            if self._requires_reshape and inspect.stack()[1][3] != 'cost':
                log.warning("You are using the `cost_from_cost_matrix` method "
                            "on a layer which has been wrapped to accept "
                            "sequence input, might or might not be "
                            "problematic.")
            return cost_from_cost_matrix(self, cost_matrix)
        return outer

    @classmethod
    def set_input_space_wrapper(cls, name, set_input_space):
        """
        If this layer is not RNN-adapted, we intercept the call to the
        set_input_space method and set the space to a non-sequence space.

        This transformation is only applied to whitelisted layers.

        Parameters
        ----------
        set_input_space : method
            The set_input_space method to be wrapped
        """
        @functools.wraps(set_input_space)
        def outer(self, input_space):
            # The set_input_space method could be called for nested MLPs
            if not self.rnn_friendly and name != 'MLP':
                def find_sequence_space(input_space):
                    """
                    Recursive helper function that searches the (possibly
                    nested) input space to see if it contains SequenceSpace
                    """
                    if isinstance(input_space, CompositeSpace):
                        return any(find_sequence_space(component) for
                                   component in input_space.components)
                    if isinstance(input_space, SequenceDataSpace):
                        return True
                    return False
                if find_sequence_space(input_space):
                    if name in BLACKLIST:
                        raise ValueError("%s received a SequenceSpace input, "
                                         "but is unable to deal with it. "
                                         "Please use an RNN-friendly "
                                         "alternative from the sandbox "
                                         "instead" % self)
                    elif name not in WHITELIST:
                        log.warning("%s received a SequenceSpace but "
                                    "is not able to deal with it. We will try "
                                    "to change to non-sequence spaces and "
                                    "reshape the data, but this is not "
                                    "guaranteed to work! It normally works if "
                                    "your input and output space are not "
                                    "nested and you are not calling other "
                                    "fprop methods from within your fprop."
                                    % self)
                    if isinstance(input_space, SequenceSpace):
                        self._requires_unmask = True
                        self._requires_reshape = True
                        input_space = input_space.space
                    elif isinstance(input_space, SequenceDataSpace):
                        self._requires_reshape = True
                        input_space = input_space.space
            return set_input_space(self, input_space)
        return outer

    @classmethod
    def get_output_space_wrapper(cls, name, get_output_space):
        """
        Same thing as set_input_space_wrapper.

        Parameters
        ----------
        get_output_space : method
            The get_output_space method to be wrapped
        """
        @functools.wraps(get_output_space)
        def outer(self):
            if (not self.rnn_friendly and self._requires_reshape and
                    not isinstance(get_output_space(self), SequenceSpace)):
                # Since not only this class, but also the Layer class gets
                # wrapped, we need to make sure we don't wrap the space
                # in a SequenceSpace twice
                return SequenceSpace(get_output_space(self))
            else:
                return get_output_space(self)
        return outer

    # Wrap get_input_space?

