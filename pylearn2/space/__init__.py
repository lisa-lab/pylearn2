import numpy as np

class Space:
    """ Defines a vector space that can be transformed by a linear operator """

    def get_origin(self):
        """ Returns the origin in this space """
        raise NotImplementedError()

    def get_origin_batch(self, n):
        """ Returns a batch of n copies of the origin """


class VectorSpace(Space):
    """ Defines a space whose points are defined as fixed-length vectors """

    def __init__(self, dim):
        """

        dim: the length of the fixed-length vector

        """

        self.dim = dim

    def get_origin(self):

        return np.zeros((self.dim,))

    def get_origin_batch(self, n):

        return np.zeros((n,self.dim))

