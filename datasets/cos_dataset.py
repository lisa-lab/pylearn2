import numpy as N
import copy

class CosDataset(object):
    """ Makes a dataset that streams randomly generated 2D examples.
        The first coordinate is sampled from a uniform distribution.
        The second coordinate is the cosine of the first coordinate,
        plus some gaussian noise. """

    def __init__(self, min_x = -6.28, max_x = 6.28, std = .05, rng = None):
        self.min_x, self.max_x, self.std = min_x, max_x, std
        if rng is None:
            rng = N.random.RandomState([17,2,946])
        #
        self.default_rng = copy.copy(rng)
        self.rng = rng
    #

    def pdf(self, mat):
        x = mat[:,0]
        y = mat[:,1]
        rval = N.exp( - ( y - N.cos(x)) ** 2. / (2. * (self.std**2.)))
        rval /= N.sqrt(2.0 * 3.1415926535 * (self.std**2.))
        rval /= (self.max_x - self.min_x)
        rval *= x < self.max_x
        rval *= x > self.min_x
        return rval


    def get_stream_position(self):
        return copy.copy(self.rng)

    def set_stream_position(self,s):
        self.rng = copy.copy(s)

    def restart_stream(self):
        self.reset_RNG()

    def reset_RNG(self):
        if 'default_rng' not in dir(self):
            self.default_rng = N.random.RandomState([17,2,946])
        self.rng = copy.copy(self.default_rng)
    #

    def apply_preprocessor(self, preprocessor, can_fit = False):
        raise NotImplementedError()
    #

    def get_batch_design(self, batch_size):
        x = self.rng.uniform(self.min_x, self.max_x, (batch_size,1))
        y = N.cos(x) + self.rng.randn(*x.shape) * self.std
        return N.hstack((x,y))
    #
#

