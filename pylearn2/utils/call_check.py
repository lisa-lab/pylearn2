"""
Utility functions for checking passed arguments against call signature
of a function or class constructor.
"""
import functools
import inspect
import types
from pylearn2.utils.string_utils import match

def check_call_arguments(to_call, kwargs):
    """
    Check the call signature against a dictionary of proposed arguments,
    raising an informative exception in the case of mismatch.

    Parameters
    ----------
    to_call : class or callable
        Function or class to examine (in the case of classes, the constructor
        call signature is analyzed).
    kwargs : dict
        Dictionary mapping parameter names (including positional arguments)
        to proposed values.
    """
    if 'self' in kwargs.keys():
        raise TypeError("Your dictionary includes an entry for 'self', "
                        "which is just asking for trouble")

    orig_to_call = getattr(to_call, '__name__', str(to_call))
    if not isinstance(to_call, types.FunctionType):
        if hasattr(to_call, '__init__'):
            to_call = to_call.__init__
        elif hasattr(to_call, '__call__'):
            to_call = to_call.__call__

    args, varargs, keywords, defaults = inspect.getargspec(to_call)

    if any(not isinstance(arg, str) for arg in args):
        raise TypeError('%s uses argument unpacking, which is deprecated and '
                        'unsupported by this pylearn2' % orig_to_call)

    if varargs is not None:
        raise TypeError('%s has a variable length argument list, but '
                        'this is not supported by config resolution' %
                        orig_to_call)

    if keywords is None:
        bad_keywords = [arg_name for arg_name in kwargs.keys()
                        if arg_name not in args]

        if len(bad_keywords) > 0:
            bad = ', '.join(bad_keywords)
            args = [ arg for arg in args if arg != 'self' ]
            if len(args) == 0:
                matched_str = '(It does not support any keywords, actually)'
            else:
                matched = [ match(keyword, args) for keyword in bad_keywords ]
                matched_str = 'Did you mean %s?' % (', '.join(matched))
            raise TypeError('%s does not support the following '
                            'keywords: %s. %s' %
                            (orig_to_call, bad, matched_str))

    if defaults is None:
        num_defaults = 0
    else:
        num_defaults = len(defaults)

    required = args[:len(args) - num_defaults]
    missing = [arg for arg in required if arg not in kwargs]

    if len(missing) > 0:
        #iff the im_self (or __self__) field is present, this is a
        # bound method, which has 'self' listed as an argument, but
        # which should not be supplied by kwargs
        is_bound = hasattr(to_call, 'im_self') or hasattr(to_call, '__self__')
        if len(missing) > 1 or missing[0] != 'self' or not is_bound:
            if 'self' in missing:
                missing.remove('self')
            missing = ', '.join([str(m) for m in missing])
            raise TypeError('%s did not get these expected '
                            'arguments: %s' % (orig_to_call, missing))

def checked_call(to_call, kwargs):
    """
    Attempt calling a function or instantiating a class with a given set of
    arguments, raising a more helpful exception in the case of argument
    mismatch.

    Parameters
    ----------
    to_call : class or callable
        Function or class to examine (in the case of classes, the constructor
        call signature is analyzed).
    kwargs : dict
        Dictionary mapping parameter names (including positional arguments)
        to proposed values.
    """
    try:
        return to_call(**kwargs)
    except TypeError:
        check_call_arguments(to_call, kwargs)
        raise

def sensible_argument_errors(func):
    """
    .. todo::

        WRITEME
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        try:
            func(*args, **kwargs)
        except TypeError:
            argnames, varargs, keywords, defaults = inspect.getargspec(func)
            posargs = dict(zip(argnames, args))
            bad_keywords = []
            for keyword in kwargs:
                if keyword not in argnames:
                    bad_keywords.append(keyword)

            if len(bad_keywords) > 0:
                bad = ', '.join(bad_keywords)
                raise TypeError('%s() does not support the following '
                                'keywords: %s' % (str(func.func_name), bad))
            allargsgot = set(list(kwargs.keys()) + list(posargs.keys()))
            numrequired = len(argnames) - len(defaults)
            diff = list(set(argnames[:numrequired]) - allargsgot)
            if len(diff) > 0:
                raise TypeError('%s() did not get required args: %s' %
                                (str(func.func_name), ', '.join(diff)))
            raise
    return wrapped_func
