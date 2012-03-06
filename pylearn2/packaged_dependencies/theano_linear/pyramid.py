import sys
import numpy
import theano
from theano.gof import Variable, Op, utils, Type, Constant,  Value, Apply
from theano.tensor import as_tensor_variable

try:
    import cv
except ImportError:
    print >> sys.stderr, "WARNING: cv not available"

def cv_available():
    return 'cv' in globals()

class GaussianPyramid(Op):
    """
    Returns `n_levels` images
    """
    default_output = slice(0,None,1) #always return a list, even when there's only one element in it
    def __init__(self, n_levels):
        self.n_levels = n_levels
    def props(self):
        return (self.n_levels,)
    def __hash__(self):
        return hash((type(self), self.props()))
    def __eq__(self, other):
        return (type(self)==type(other) and self.props() == other.props())
    def __repr__(self):
        return '%s{n_levels=%s}' %(self.__class__.__name__, self.n_levels)
    def infer_shape(self, node, input_shapes):
        xshp, = input_shapes
        out_shapes = [xshp]
        while len(out_shapes) < self.n_levels:
            s = out_shapes[-1]
            out_shapes.append((s[0], s[1]//2, s[2]//2,s[3]))
        return out_shapes
    def make_node(self, x):
        if self.n_levels < 1:
            raise ValueError(('It does not make sense for'
                ' GaussianPyramid to generate %i levels'),
                self.n_levels)
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type() for i in range(self.n_levels)])
    def perform(self, node, ins, outs):
        x, = ins
        outs[0][0] = z = x.copy()
        B,M,N,K = x.shape
        for level in range(1,self.n_levels):
            # z is the whole pyramid at level `level-1`
            # loop body builds `out` which is the pyramid at `level`
            z0 = z[0]
            if z0.shape[0] <=2 or z0.shape[1] <= 2:
                raise ValueError('Cannot downsample an image smaller than 3x3',
                        z0.shape)
            print z0.shape, z0.dtype, z0.strides
            out0 = cv.pyrDown(z0)
            assert out0.dtype == x.dtype
            if out0.ndim ==3:
                assert out0.shape[2] == x.shape[3] # assert same # channels
            else:
                assert K==1
            out = numpy.empty(
                    (x.shape[0],
                        out0.shape[0],
                        out0.shape[1],
                        K),
                    dtype=out0.dtype)
            if K==1:
                out[0][:,:,0] = out0
            else:
                out[0] = out0
            for i, zi in enumerate(z[1:]):
                if K==1:
                    out[i][:,:,0] = cv.pyrDown(z[i])
                else:
                    out[i] = cv.pyrDown(z[i])
            outs[level][0] = out
            z = out


# test infer shape

# test non power-of-two shapes

# test different numbers of channels

def test_gaussian_pyramid_shapes():
    for dtype in ('float32', 'float64'):
        x = theano.tensor.tensor4(dtype=dtype)
        f = theano.function([x], GaussianPyramid(3)(x))

        xval = numpy.ones((1, 64, 64, 1), dtype=dtype)
        a,b,c = f(xval)
        assert a.shape == (1,64,64,1)
        assert b.shape == (1,32,32,1)
        assert c.shape == (1,16,16,1)

        xval = numpy.ones((1, 12, 12, 10), dtype=dtype)
        a,b,c = f(xval)
        assert a.shape == (1,12,12,10)
        assert b.shape == (1,6,6,10)
        assert c.shape == (1,3,3,10)

        p = GaussianPyramid(1)(x)
        f = theano.function([x], p)
        a, = f(xval)
        assert a.shape  == xval.shape
        #print a.max(), a.min()
        #print x.max(), x.min()
        #assert numpy.allclose(a, x)
