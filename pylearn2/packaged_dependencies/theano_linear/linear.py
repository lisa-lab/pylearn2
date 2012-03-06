"""
XXX
"""
import sys
import copy
import numpy
import theano

prod = numpy.prod

def dot(x, y):
    """Return the linear transformation of `y` by `x` or `x` by `y` when one
    or both of `x` and `y` is a LinearTransform instance
    """
    if isinstance(x, LinearTransform):
        return x.rmul(y)
    elif isinstance(y, LinearTransform):
        return y.lmul(x)
    else:
        return theano.dot(x,y)


def dot_shape_from_shape(x, y):
    """Compute `dot(x, y).shape` from the shape of the non-LinearTransform
    """
    if isinstance(x, LinearTransform):
        if type(y) != tuple:
            raise TypeError('y should be tuple', y)
        return x.col_shape() + x.split_right_shape(y, False)[1]
    elif isinstance(y, LinearTransform):
        if type(x) != tuple:
            raise TypeError('x should be tuple', x)
        return y.split_left_shape(x, False)[0] + y.row_shape()
    else:
        raise TypeError('One of x or y should be a LinearTransform')


def dot_shape(x, y):
    """Return the linear transformation of `y` by `x` or `x` by `y` when one
    or both of `x` and `y` is a LinearTransform instance
    """
    if isinstance(x, LinearTransform):
        return dot_shape_from_shape(x, tuple(y.shape))
    elif isinstance(y, LinearTransform):
        return dot_shape_from_shape(tuple(x.shape), y)
    else:
        raise TypeError('One of x or y should be a LinearTransform')


class LinearTransform(object):
    """

    Attributes:
    _params: a list of theano shared variables
        that parametrize the linear transformation

    """
    def __init__(self, params):
        self.set_params(params)

    def set_params(self, params):
        self._params = list(params)

    def params(self):
        return list(self._params)

    def __str__(self):
        return self.__class__.__name__ +'{}'

    # N.B. Don't implement __mul__ and __lmul__ because these mean
    # element-wise multiplication in numpy land.

    def __add__(self, other):
        return Sum([self, other])

    def __radd__(self, other):
        return Sum([other, self])

    # OVER-RIDE THIS (or rmul)
    def lmul(self, x):
        """mul(x, A)

        """
        # this is a circular definition with rmul so that they are both
        # implemented as soon as one of them is overridden by a base class.

        try:
            # dot(x, A)
            # = dot(A.T, x.T).T
            AT_xT = self.rmul_T(self.transpose_left(x, False))
            rval = self.transpose_right(AT_xT, True)
            return rval
        except RuntimeError, e:
            if 'ecursion' in str(e):
                raise TypeError('either lmul or rmul_T must be implemented')
            raise
        except TypeError, e:
            if 'either lmul' in str(e):
                raise TypeError('either lmul or rmul_T must be implemented')

    def lmul_T(self, x):
        # this is a circular definition with rmul so that they are both
        # implemented as soon as one of them is overridden by a base class.

        # dot(x, A.T)
        # = dot(A, x.T).T
        A_xT = self.rmul(self.transpose_right(x, True))
        rval = self.transpose_left(A_xT, True)
        return rval

    # OVER-RIDE THIS (or lmul)
    def rmul(self, x):
        """mul(A, x) or mul(A.T, x)
        """
        # this is a circular definition with rmul so that they are both
        # implemented as soon as one of them is overridden by a base class.

        try:
            # dot(A, x)
            # = dot(x.T, A.T).T
            xT_AT = self.lmul_T(self.transpose_right(x, False))
            rval = self.transpose_left(xT_AT, False)
            return rval
        except RuntimeError, e:
            if 'ecursion' in str(e):
                raise TypeError('either rmul or lmul_T must be implemented')
            raise
        except TypeError, e:
            if 'either lmul' in str(e):
                raise TypeError('either rmul or lmul_T must be implemented')

    def rmul_T(self, x):
        # this is a circular definition with rmul so that they are both
        # implemented as soon as one of them is overridden by a base class.

        # dot (A.T, x)
        # = dot(x.T, A).T
        xT_A = self.lmul(self.transpose_left(x, True))
        rval = self.transpose_right(xT_A, True)
        return rval

    def transpose_left(self, x, T):
        """
        XXX
        """
        # supposing self.row_shape is (R1,)...
        xshp = tuple(x.shape)
        cshp = self.col_shape()
        if T:
            # C1 C2 C3 R1 R2 -> R1 R2 C1 C2 C3
            ss = len(cshp)
        else:
            # R1 R2 C1 C2 C3 -> C1 C2 C3 R1 R2
            ss = x.ndim - len(cshp)
        pattern = range(ss, x.ndim) + range(ss)
        return x.transpose(pattern)

    def transpose_right(self, x, T):
        """
        XXX
        """
        # supposing self.row_shape is (R1,)...
        xshp = tuple(x.shape)
        rshp = self.row_shape()
        if T:
            # C1 C2 R1 -> R1 C1 C2
            ss = len(rshp)
        else:
            # R1 C1 C2 -> C1 C2 R1
            ss = x.ndim - len(rshp)
        pattern = range(ss, x.ndim) + range(ss)
        return x.transpose(pattern)

    def split_left_shape(self, xshp, T):
        """
        XXX
        """
        if type(xshp) != tuple:
            raise TypeError('need tuple', xshp)
        # supposing self.col_shape is (C1, C2, C3) ...
        cshp = self.col_shape()
        assert type(cshp) == tuple
        if T:
            # C1 C2 C3 R1 R2
            ss = len(cshp)
            RR, CC = xshp[ss:], xshp[:ss]
        else:
            # R1 R2 C1 C2 C3
            ss = len(xshp) - len(cshp)
            RR, CC = xshp[:ss], xshp[ss:]
        if len(CC) != len(cshp) or (
                not all((isinstance(cc, theano.Variable) or cc == ci)
                    for cc, ci in zip(CC, cshp))):
            raise ValueError('invalid left shape',
                    dict(xshp=xshp, col_shape=cshp, xcols=CC, T=T))
        if T:
            return CC, RR
        else:
            return RR, CC

    def split_right_shape(self, xshp, T):
        """
        XXX
        """
        if type(xshp) != tuple:
            raise TypeError('need tuple', xshp)
        # supposing self.row_shape is (R1, R2) ...
        rshp = self.row_shape()
        assert type(rshp) == tuple
        if T:
            # C1 C2 C3 R1 R2
            ss = len(xshp) - len(rshp)
            RR, CC = xshp[ss:], xshp[:ss]
        else:
            # R1 R2 C1 C2 C3
            ss = len(rshp)
            RR, CC = xshp[:ss], xshp[ss:]
        if len(RR) != len(rshp) or (
                not all((isinstance(rr, theano.Variable) or rr == ri)
                    for rr, ri in zip(RR, rshp))):
            raise ValueError('invalid left shape',
                    dict(xshp=xshp, row_shape=rshp, xrows=RR, T=T))
        if T:
            return CC, RR
        else:
            return RR, CC

    def transpose_left_shape(self, xshp, T):
        """
        """
        RR, CC = self.split_left_shape(xshp, T)
        return CC + RR

    def transpose_right_shape(self, xshp, T):
        """
        """
        RR, CC = self.split_right_shape(xshp, T)
        return CC + RR

    def is_valid_left_shape(self, xshp, T):
        """
        """
        try:
            self.split_left_shape(xshp, T)
            return True
        except ValueError:
            return False

    def is_valid_right_shape(self, xshp, T):
        """
        """
        try:
            self.split_right_shape(xshp, T)
            return True
        except ValueError:
            return False

    # OVER-RIDE THIS
    def row_shape(self):
        raise NotImplementedError('override me')

    # OVER-RIDE THIS
    def col_shape(self):
        raise NotImplementedError('override me')

    def transpose(self):
        return TransposeTransform(self)

    T = property(lambda self: self.transpose())

    # OVER-RIDE THIS
    def tile_columns(self, **kwargs):
        raise NotImplementedError('override me')


class TransposeTransform(LinearTransform):
    def __init__(self, base):
        super(TransposeTransform, self).__init__([])
        self.base = base

    def transpose(self):
        return self.base

    def params(self):
        return self.base.params()

    def lmul(self, x):
        return self.base.lmul_T(x)

    def lmul_T(self, x):
        return self.base.lmul(x)

    def rmul(self, x):
        return self.base.rmul_T(x)

    def rmul_T(self, x):
        return self.base.rmul(x)

    def transpose_left(self, x, T):
        return self.base.transpose_right(x, not T)

    def transpose_right(self, x, T):
        return self.base.transpose_left(x, not T)

    def transpose_left_shape(self, x, T):
        return self.base.transpose_right_shape(x, not T)

    def transpose_right_shape(self, x, T):
        return self.base.transpose_left_shape(x, not T)

    def split_left_shape(self, x, T):
        return self.base.split_right_shape(x, not T)

    def split_right_shape(self, x, T):
        return self.base.split_left_shape(x, not T)

    def is_valid_left_shape(self, x, T):
        return self.base.is_valid_right_shape(x, not T)

    def is_valid_right_shape(self, x, T):
        return self.base.is_valid_left_shape(x, not T)

    def row_shape(self):
        return self.base.col_shape()

    def col_shape(self):
        return self.base.row_shape()

    def print_status(self):
        return self.base.print_status()

    def tile_columns(self):
        # yes, it would be nice to do rows, but since this is a visualization
        # and there *is* no tile_rows, we fall back on this.
        return self.base.tile_columns()


if 0: # needs to be brought up to date with LinearTransform method names
    class Concat(LinearTransform):
        """
        Form a linear map of the form [A B ... Z].

        For this to be valid, A,B...Z must have identical row_shape.

        The col_shape defaults to being the concatenation of flattened output from
        each of A,B,...Z, but a col_shape tuple specified via the constructor will
        reshape that vector.
        """
        def __init__(self, Wlist, col_shape=None):
            super(Concat, self).__init__([])
            self._Wlist = list(Wlist)
            if not isinstance(col_shape, (int,tuple,type(None))):
                raise TypeError('col_shape must be int or int tuple')
            self._col_sizes = [prod(w.col_shape()) for w in Wlist]
            if col_shape is None:
                self.__col_shape = sum(self._col_sizes),
            elif isinstance(col_shape, int):
                self.__col_shape = col_shape,
            else:
                self.__col_shape = tuple(col_shape)
            assert prod(self.__col_shape) == sum(self._col_sizes)
            self.__row_shape = Wlist[0].row_shape()
            for W in Wlist[1:]:
                if W.row_shape() != self.row_shape():
                    raise ValueError('Transforms has different row_shape',
                            W.row_shape())

        def params(self):
            rval = []
            for W in self._Wlist:
                rval.extend(W.params())
            return rval
        def _lmul(self, x, T):
            if T:
                if len(self.col_shape())>1:
                    x2 = x.flatten(2)
                else:
                    x2 = x
                n_rows = x2.shape[0]
                offset = 0
                xWlist = []
                assert len(self._col_sizes) == len(self._Wlist)
                for size, W in zip(self._col_sizes, self._Wlist):
                    # split the output rows into pieces
                    x_s = x2[:,offset:offset+size]
                    # multiply each piece by one transform
                    xWlist.append(
                            W.lmul(
                                x_s.reshape(
                                    (n_rows,)+W.col_shape()),
                                T))
                    offset += size
                # sum the results
                rval = tensor.add(*xWlist)
            else:
                # multiply the input by each transform
                xWlist = [W.lmul(x,T).flatten(2) for W in self._Wlist]
                # join the resuls
                rval = tensor.join(1, *xWlist)
            return rval
        def _col_shape(self):
            return self.__col_shape
        def _row_shape(self):
            return self.__row_shape
        def _tile_columns(self):
            # hard-coded to produce RGB images
            arrays = [W._tile_columns() for W in self._Wlist]
            o_rows = sum([a.shape[0]+10 for a in arrays]) - 10
            o_cols = max([a.shape[1] for a in arrays])
            rval = numpy.zeros(
                    (o_rows, o_cols, 3),
                    dtype=arrays[0].dtype)
            offset = 0
            for a in arrays:
                if a.ndim==2:
                    a = a[:,:,None] #make greyscale broadcast over colors
                rval[offset:offset+a.shape[0], 0:a.shape[1],:] = a
                offset += a.shape[0] + 10
            return rval
        def print_status(self):
            for W in self._Wlist:
                W.print_status()


if 0: # needs to be brought up to date with LinearTransform method names
    class Sum(LinearTransform):
        def __init__(self, terms):
            self.terms = terms
            for t in terms[1:]:
                assert t.row_shape() == terms[0].row_shape()
                assert t.col_shape() == terms[0].col_shape()
        def params(self):
            rval = []
            for t in self.terms:
                rval.extend(t.params())
            return rval
        def _lmul(self, x, T):
            results = [t._lmul(x, T)]
            return tensor.add(*results)
        def _row_shape(self):
            return self.terms[0].col_shape()
        def _col_shape(self):
            return self.terms[0].row_shape()
        def print_status(self):
            for t in terms:
                t.print_status()
        def _tile_columns(self):
            raise NotImplementedError('TODO')


if 0: # This is incomplete
    class Compose(LinearTransform):
        """ For linear transformations [A,B,C]
        this represents the linear transformation A(B(C(x))).
        """
        def __init__(self, linear_transformations):
            self._linear_transformations = linear_transformations
        def dot(self, x):
            return reduce(
                    lambda t,a:t.dot(a),
                    self._linear_transformations,
                    x)
        def transpose_dot(self, x):
            return reduce(
                    lambda t, a: t.transpose_dot(a),
                    reversed(self._linear_transformations),
                    x)
        def params(self):
            return reduce(
                    lambda t, a: a + t.params(),
                    self._linear_transformations,
                    [])

