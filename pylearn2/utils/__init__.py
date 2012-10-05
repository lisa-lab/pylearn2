import warnings

from .general import is_iterable
import theano

import numpy

def make_name(variable, anon = "anonymous_variable"):
    """
    If variable has a name, returns that name.
    Otherwise, returns anon
    """

    if hasattr(variable,'name') and variable.name is not None:
        return variable.name

    return anon


def sharedX(value, name=None, borrow=False):
    """Transform value into a shared variable of type floatX"""
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
         name=name,
         borrow=borrow)

def as_floatX(variable):
    """Casts a given variable into dtype config.floatX
    numpy ndarrays will remain numpy ndarrays
    python floats will become 0-D ndarrays
    all other types will be treated as theano tensors"""

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def constantX(value):
    """
        Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value,
                                     dtype=theano.config.floatX))
def subdict(d, keys):
    """ Create a subdictionary of d with the keys in keys """
    result = {}
    for key in keys:
        if key in d: result[key] = d[key]
    return result

def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


class CallbackOp(theano.gof.op):
    """A Theano Op that implements the identity transform but
    also does an arbitrary (user-specified) side effect. """


    view_map = { 0: [0] }

    def __init__(self, callback):
        self.callback = callback

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.callback(xin)

    def grad(self, inputs, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return type(self) == type(other) and self.callback == other.callback

    def hash(self):
        return hash(self.callback)

