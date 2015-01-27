import theano
theano.config.compute_test_value = 'off'
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
import numpy as N
from theano.compat.six.moves import xrange
import theano.tensor as T
from theano import function
from pylearn2.utils import as_floatX
from pylearn2.utils import sharedX
from pylearn2.linear.matrixmul import MatrixMul
import unittest


class TestGRBM_Type_1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_m = 2

        cls.rng = N.random.RandomState([1, 2, 3])
        cls.nv = 3
        cls.nh = 4

        cls.vW = cls.rng.randn(cls.nv, cls.nh)
        cls.W = sharedX(cls.vW)
        cls.vbv = as_floatX(cls.rng.randn(cls.nv))
        cls.bv = T.as_tensor_variable(cls.vbv)
        cls.bv.tag.test_value = cls.vbv
        cls.vbh = as_floatX(cls.rng.randn(cls.nh))
        cls.bh = T.as_tensor_variable(cls.vbh)
        cls.bh.tag.test_value = cls.bh
        cls.vsigma = as_floatX(cls.rng.uniform(0.1, 5))
        cls.sigma = T.as_tensor_variable(cls.vsigma)
        cls.sigma.tag.test_value = cls.vsigma

        cls.E = GRBM_Type_1(transformer=MatrixMul(cls.W), bias_vis=cls.bv,
                            bias_hid=cls.bh, sigma=cls.sigma)

        cls.V = T.matrix()
        cls.V.tag.test_value = as_floatX(cls.rng.rand(cls.test_m, cls.nv))
        cls.H = T.matrix()
        cls.H.tag.test_value = as_floatX(cls.rng.rand(cls.test_m, cls.nh))

        cls.E_func = function([cls.V, cls.H], cls.E([cls.V, cls.H]))
        cls.F_func = function([cls.V], cls.E.free_energy(cls.V))
        cls.log_P_H_given_V_func = \
            function([cls.H, cls.V], cls.E.log_P_H_given_V(cls.H, cls.V))
        cls.score_func = function([cls.V], cls.E.score(cls.V))

        cls.F_of_V = cls.E.free_energy(cls.V)
        cls.dummy = T.sum(cls.F_of_V)
        cls.negscore = T.grad(cls.dummy, cls.V)
        cls.score = - cls.negscore

        cls.generic_score_func = function([cls.V], cls.score)

    def test_mean_H_given_V(self):
        tol = 1e-6

        # P(h_1 | v) / P(h_2 | v) = a
        # => exp(-E(v, h_1)) / exp(-E(v,h_2)) = a
        # => exp(E(v,h_2)-E(v,h_1)) = a
        # E(v,h_2) - E(v,h_1) = log(a)
        # also log P(h_1 | v) - log P(h_2) = log(a)

        rng = N.random.RandomState([1, 2, 3])

        m = 5

        Vv = as_floatX(N.zeros((m, self.nv)) + rng.randn(self.nv))

        Hv = as_floatX(rng.randn(m, self.nh) > 0.)

        log_Pv = self.log_P_H_given_V_func(Hv, Vv)

        Ev = self.E_func(Vv, Hv)

        for i in xrange(m):
            for j in xrange(i + 1, m):
                log_a = log_Pv[i] - log_Pv[j]
                e = Ev[j] - Ev[i]

                assert abs(e-log_a) < tol

    def test_free_energy(self):

        rng = N.random.RandomState([1, 2, 3])

        m = 2 ** self.nh

        Vv = as_floatX(N.zeros((m, self.nv)) + rng.randn(self.nv))

        F, = self.F_func(Vv[0:1, :])

        Hv = as_floatX(N.zeros((m, self.nh)))

        for i in xrange(m):
            for j in xrange(self.nh):
                Hv[i, j] = (i & (2 ** j)) / (2 ** j)

        Ev = self.E_func(Vv, Hv)

        Fv = -N.log(N.exp(-Ev).sum())
        assert abs(F-Fv) < 1e-6

    def test_score(self):
        rng = N.random.RandomState([1, 2, 3])

        m = 10

        Vv = as_floatX(rng.randn(m, self.nv))

        Sv = self.score_func(Vv)
        gSv = self.generic_score_func(Vv)

        assert N.allclose(Sv, gSv)
