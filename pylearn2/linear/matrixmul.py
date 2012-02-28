from theano_linear.matrixmul import MatrixMul as OrigMatrixMul

class MatrixMul(OrigMatrixMul):

    def get_params(self):
        return set([self._W])
