import theano
theano.config.compute_test_value = 'off'
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
import numpy as N
import theano.tensor as T
from theano import function
from pylearn2.utils import as_floatX
from pylearn2.utils import sharedX
from pylearn2.linear.matrixmul import MatrixMul

test_m = 2

rng = N.random.RandomState([1,2,3])
nv = 3
nh = 4

vW = rng.randn(nv,nh)
W = sharedX(vW)
vbv = rng.randn(nv)
bv = T.as_tensor_variable(vbv)
bv.tag.test_value = vbv
vbh = rng.randn(nh)
bh = T.as_tensor_variable(vbh)
bh.tag.test_value = bh
vsigma = rng.uniform(0.1,5)
sigma = T.as_tensor_variable(vsigma)
sigma.tag.test_value = vsigma

E = GRBM_Type_1(transformer = MatrixMul(W), bias_vis = bv, bias_hid = bh, sigma = sigma)

V = T.matrix()
V.tag.test_value = rng.rand(test_m,nv)
H = T.matrix()
H.tag.test_value = rng.rand(test_m,nh)

E_func = function([V,H],E([V,H]))
F_func = function([V],E.free_energy(V))
log_P_H_given_V_func = function([H,V],E.log_P_H_given_V(H,V))
score_func = function([V],E.score(V))

F_of_V = E.free_energy(V)
dummy = T.sum(F_of_V)
negscore = T.grad(dummy,V)
score = - negscore

generic_score_func = function([V],score)

class TestGRBM_Type_1:
    def test_mean_H_given_V(self):
        tol = 1e-6

        # P(h_1 | v) / P(h_2 | v) = a
        # => exp(-E(v, h_1)) / exp(-E(v,h_2)) = a
        # => exp(E(v,h_2)-E(v,h_1)) = a
        # E(v,h_2) - E(v,h_1) = log(a)
        # also log P(h_1 | v) - log P(h_2) = log(a)

        rng = N.random.RandomState([1,2,3])

        m = 5

        Vv = as_floatX(N.zeros((m,nv))+rng.randn(nv))

        Hv = as_floatX(rng.randn(m,nh) > 0.)

        log_Pv = log_P_H_given_V_func(Hv,Vv)

        Ev = E_func(Vv,Hv)

        for i in xrange(m):
            for j in xrange(i+1,m):
                log_a = log_Pv[i] - log_Pv[j]
                e = Ev[j] - Ev[i]

                assert abs(e-log_a) < tol



    def test_free_energy(self):

        rng = N.random.RandomState([1,2,3])

        m = 2 ** nh

        Vv = as_floatX(N.zeros((m,nv))+rng.randn(nv))


        F ,= F_func(Vv[0:1,:])

        Hv = as_floatX(N.zeros((m,nh)))

        for i in xrange(m):
            for j in xrange(nh):
                Hv[i,j] = (i & (2 ** j)) / ( 2 ** j)


        Ev = E_func(Vv,Hv)

        Fv = -N.log(N.exp(-Ev).sum())
        assert abs(F-Fv) < 1e-6


    def test_score(self):
        rng = N.random.RandomState([1,2,3])

        m = 10

        Vv = as_floatX(rng.randn(m,nv))

        Sv = score_func(Vv)
        gSv = generic_score_func(Vv)


        assert N.allclose(Sv,gSv)

