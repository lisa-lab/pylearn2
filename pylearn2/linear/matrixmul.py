from pylearn2.packaged_dependencies.theano_linear.matrixmul import MatrixMul as OrigMatrixMul
from pylearn2.linear.linear_transform import LinearTransform as PL2LT
import functools

class MatrixMul(OrigMatrixMul):
    """ The most basic LinearTransform: matrix multiplication. See TheanoLinear
    for more documentation. """

    @functools.wraps(PL2LT.get_params)
    def get_params(self):
        return set([self._W])
