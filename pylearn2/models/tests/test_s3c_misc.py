import warnings
from theano.sandbox.linalg.ops import alloc_diag
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from pylearn2.models.s3c import E_Step
from pylearn2.models.s3c import Grad_M_Step
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
from theano import config
from pylearn2.utils import serial

if config.floatX != 'float64':
    warnings.warn("Changing floatX to float64, unsure if these tests work for float32 yet")
    config.floatX = 'float64'


class TestS3C_Misc:

    def __init__(self):
        """ gets a small batch of data
            sets up an S3C model and learns on the data
            creates an expression for the log likelihood of the data
        """

        self.tol = 1e-5

        #dataset = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_train_1K.pkl')

        X = np.random.RandomState([1,2,3]).randn(1000,108)
        #dataset.get_batch_design(1000)
        #X = X[:,0:2]
        #warnings.warn('hack')
        #X[0,0] = 1.
        #X[0,1] = -1.
        m, D = X.shape
        N = 300

        self.model = S3C(nvis = D,
                #disable_W_update = 1,
                         nhid = N,
                         irange = .5,
                         init_bias_hid = 0.,
                         init_B = 1.,
                         min_B = 1e-8,
                         max_B = 1e8,
                         tied_B = 1,
                         e_step = E_Step(
                             #h_new_coeff_schedule = [ ],
                             h_new_coeff_schedule = [ .01 ]
                         ),
                         init_alpha = 1.,
                         min_alpha = 1e-8, max_alpha = 1e8,
                         init_mu = 1.,
                         m_step = Grad_M_Step( learning_rate = 1.0 ),
                        )

        #warnings.warn('hack')
        #W = self.model.W.get_value()
        #W[0,0] = 1.
        #W[1,0] = 1.
        #self.model.W.set_value(W)

        self.orig_params = self.model.get_param_values()

        model = self.model
        self.mf_obs = model.e_step.variational_inference(X)

        self.stats = SufficientStatistics.from_observations(needed_stats =
                model.m_step.needed_stats(), V =X,
                ** self.mf_obs)

        self.prob = self.model.expected_log_prob_vhs( self.stats , H_hat = self.mf_obs['H_hat'], S_hat = self.mf_obs['S_hat'])
        self.X = X


    def test_expected_log_prob_vhs_batch_match(self):
        """ verifies that expected_log_prob_vhs = mean(expected_log_prob_vhs_batch) """

        scalar = self.model.expected_log_prob_vhs( stats = self.stats, H_hat = self.mf_obs['H_hat'], S_hat = self.mf_obs['S_hat'])
        batch  = self.model.expected_log_prob_vhs_batch( V = self.X, H_hat = self.mf_obs['H_hat'], S_hat = self.mf_obs['S_hat'], var_s0_hat = self.mf_obs['var_s0_hat'], var_s1_hat = self.mf_obs['var_s1_hat'])

        f = function([], [scalar, batch] )

        res1, res2 = f()

        res2 = res2.mean(dtype='float64')

        print res1, res2

        assert np.allclose(res1, res2)



    def test_grad_alpha(self):
        """tests that the gradient of the log probability with respect to alpha
        matches my analytical derivation """

        #self.model.set_param_values(self.new_params)

        g = T.grad(self.prob, self.model.alpha, consider_constant = self.mf_obs.values())

        mu = self.model.mu
        alpha = self.model.alpha
        half = as_floatX(.5)

        mean_sq_s = self.stats.d['mean_sq_s']
        mean_hs = self.stats.d['mean_hs']
        mean_h = self.stats.d['mean_h']

        term1 = - half * mean_sq_s

        term2 = mu * mean_hs

        term3 = - half * T.sqr(mu) * mean_h

        term4 = half / alpha

        analytical = term1 + term2 + term3 + term4

        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > self.tol:
            print "gv"
            print gv
            print "av"
            print av
            raise Exception("analytical gradient on alpha deviates from theano gradient on alpha by up to "+str(max_diff))

    def test_grad_W(self):
        """tests that the gradient of the log probability with respect to W
        matches my analytical derivation """

        #self.model.set_param_values(self.new_params)

        g = T.grad(self.prob, self.model.W, consider_constant = self.mf_obs.values())

        B = self.model.B
        W = self.model.W
        mean_hsv = self.stats.d['mean_hsv']

        mean_sq_hs = self.stats.d['mean_sq_hs']

        mean_HS = self.mf_obs['H_hat'] * self.mf_obs['S_hat']

        m = mean_HS.shape[0]

        outer_prod = T.dot(mean_HS.T,mean_HS)
        outer_prod.name = 'outer_prod<from_observations>'
        outer = outer_prod/m
        mask = T.identity_like(outer)
        second_hs = (1.-mask) * outer + alloc_diag(mean_sq_hs)


        term1 = (B * mean_hsv).T
        term2 = - B.dimshuffle(0,'x') * T.dot(W, second_hs)

        analytical = term1 + term2

        f = function([],(g,analytical))

        gv, av = f()

        assert gv.shape == av.shape

        max_diff = np.abs(gv-av).max()

        if max_diff > self.tol:
            print "gv"
            print gv
            print "av"
            print av
            raise Exception("analytical gradient on W deviates from theano gradient on W by up to "+str(max_diff))

if __name__ == '__main__':
    obj = TestS3C_Misc()
    obj.test_grad_W()
