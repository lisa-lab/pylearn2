"""
Documentation-related helper classes/functions
"""


class soft_wraps:
    """
    A Python decorator which concatenates two functions' docstrings: one
    function is defined at initialization and the other one is defined when
    soft_wraps is called.

    This helps reduce the ammount of documentation to write: one can use
    this decorator on child classes' functions when their implementation is
    similar to the one of the parent class. Conversely, if a function defined
    in a child class departs from its parent's implementation, one can simply
    explain the differences in a 'Notes' section without re-writing the whole
    docstring.

    Examples
    --------
    >>> class Parent(object):
    ...     def f(x):
    ...        '''
    ...        Adds 1 to x
    ...        
    ...        Parameters
    ...        ----------
    ...        x : int
    ...            Variable to increment by 1
    ...
    ...        Returns
    ...        -------
    ...        rval : int
    ...            x incremented by 1
    ...        '''
    ...        rval = x + 1
    ...        return rval
    ...
    >>> class Child(Parent):
    ...     @soft_wraps(Parent.f)
    ...     def f(x):
    ...        '''
    ...        Notes
    ...        -----
    ...        Also prints the incremented value
    ...        '''
    ...        rval = x + 1
    ...        print rval
    ...        return rval
    ...
    >>> c = Child()
    >>> print c.f.__doc__

        Adds 1 to x
        
        Parameters
        ----------
        x : int
            Variable to increment by 1
    
        Returns
        -------
        rval : int
           x incremented by 1
    
        Notes
        -----
        Also prints the incremented value
    """

    def __init__(self, f, append=False):
        """
        Parameters
        ----------
        f : function
            Function whose docstring will be concatenated with the decorated
            function's docstring
        prepend : bool, optional
            If True, appends f's docstring to the decorated function's
            docstring instead of prepending it. Defaults to False.
        """
        self.f = f
        self.append = append

    def __call__(self, f):
        """
        Prepend self.f's docstring to f's docstring (or append it if
        `self.append == True`).

        Parameters
        ----------
        f : function
            Function to decorate

        Returns
        -------
        f : function
            Function f passed as argument with self.f's docstring
            {pre,ap}pended to it
        """
        if self.append:
            f.__doc__ +=  + self.f.__doc__
        else:
            f.__doc__ = self.f.__doc__ + f.__doc__

        return f
