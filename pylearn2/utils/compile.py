"""Utilities related to the compilation of Theano functions."""
import functools

__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, David Warde-Farley / Universite de Montreal"
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"
__all__ = ["compiled_theano_function", "HasCompiledFunctions"]


def compiled_theano_function(fn):
    """
    Method decorator that enables lazy on-demand compilation of Theano
    functions.

    Parameters
    ----------
    fn : bound method
        Method that takes exactly one parameter (i.e. `self`). This method
        should return a compiled Theano function when called.

    Notes
    -----
    This will return an object property (i.e. the `property()` construct)
    that returns a Theano function, and store it in a dictionary attribute
    called `_compiled_functions` on the object. If the function has
    already been accessed it is not compiled again.

    Objects wishing to take advantage of this decorator should inherit
    from `HasCompiledFunctions` in this module to have the object's
    `_compiled_functions` attribute removed upon pickling.

    Examples
    --------
    >>> from pylearn2.utils.compile import compiled_theano
    >>> import theano
    >>> class Foo(object):
    ...     @compiled_theano_function
    ...     def bar(self):
    ...         x = theano.tensor.vector()
    ...         y = theano.tensor.vector()
    ...         print "Compiling..."
    ...         return theano.function([x, y], theano.tensor.dot(x, y))
    ...
    >>> from numpy.random import randn, seed
    >>> o = Foo()
    >>> seed(0)
    >>> xx, yy = randn(50), randn(50)
    >>> o.bar(xx, yy)  # first call, method body will be run
    Compiling...
    array(-3.0349256483108418)
    >>> o.bar(xx, yy)  # function already compiled, no print.
    array(-3.0349256483108418)
    >>> o.bar(randn(5), randn(5))  # different args, still no print
    array(5.294487561729036)
    """
    @functools.wraps(fn)
    def wrapped(self):
        try:
            func = self._compiled_functions[fn.func_name]
        except (AttributeError, KeyError):
            if not hasattr(self, '_compiled_functions'):
                self._compiled_functions = {}
            self._compiled_functions[fn.func_name] = func = fn(self)
        return func
    return property(wrapped)


class HasCompiledFunctions(object):
    """
    Base class/mixin that will automatically strip a `_compiled_functions`
    attribute when pickling.
    """
    def __getstate__(self):
        """
        .. todo::

            WRITEME
        """
        state = self.__dict__.copy()
        if '_compiled_functions' in state:
            del state['_compiled_functions']
        return state
