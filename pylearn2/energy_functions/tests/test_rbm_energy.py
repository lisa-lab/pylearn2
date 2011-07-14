from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
import numpy as N
import theano.tensor as T
from theano import function

rng = N.random.RandomState([1,2,3])
nv = 3
nh = 4

W = rng.randn(nv,nh)
bv = rng.randn(nv)
bh = rng.randn(nh)
sigma = rng.uniform(0.1,5)

E = GRBM_Type_1(W = W, bias_vis = bv, bias_hid = bh, sigma = sigma)

V = T.matrix()
H = T.matrix()

E_func = function([V,H],E([V,H]))
F_func = function([V],E.free_energy(V))
mean_H_given_V_func = function([V],E.mean_H_given_V(V))

class TestGRBM_Type_1:
    def test_mean_H_given_V(self):
        tol = 1e-6

        # P(h_1 | v) / P(h_2 | v) = a
        # => exp(-E(v, h_1)) / exp(-E(v,h_2)) = a
        # => exp(E(v,h_2)-E(v,h_1)) = a
        # E(v,h_2) - E(v,h_1) = log(a)

        rng = N.random.RandomState([1,2,3])

        m = 5

        Vv = rng.randn(m,nv)

        Hv = mean_H_given_V_func(Vv)

        Ev = E_func(Vv)

        for i in xrange(m):
            for j in xrange(i+1,m):
                a = Hv[i] / Hv[j]
                log_a = N.log(a)
                e = Ev[j] - Ev[i]

                assert abs(e-log_a) < tol
