"""
Sandbox projection operator for natural language processing (NLP)
    os.path.join(pylearn2.__path__[0],
                 "sandbox/nlp/scripts/language_model/language_model.yaml")
"""
from pylearn2.linear import matrixmul


class MatrixMul(matrixmul.MatrixMul):
    """
    Operations which can be represented as matrix multiplications.
    """
    def init(self):
        """
        Call the parent constructor
        """
        super(MatrixMul, self).__init__()

    def project(self, x):
        """
        Takes a sequence of integers and projects (embeds) these labels
        into a continuous space by concatenating the correspending
        rows in the projection matrix W i.e. [2, 5] -> [W[2] ... W[5]]

        Parameters
        ----------
        x : theano.tensor, int dtype
            A vector of labels (or a matrix where each row is a sample in
            a batch) which will be projected
        """
        if x.ndim == 2:
            shape = (x.shape[0], x.shape[1] * self._W.shape[1])
            return self._W[x.flatten()].reshape(shape)
        elif x.ndim == 1:
            return self._W[x].reshape((x.shape[0] * self._W.shape[1],))
        else:
            assert ValueError("project needs 1- or 2-dimensional input")
