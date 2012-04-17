import warnings
from theano.sandbox.linalg.ops import alloc_diag
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from pylearn2.models.s3c import E_Step_Scan
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
                         init_bias_hid = -.1,
                         init_B = 1.,
                         min_B = 1e-8,
                         max_B = 1e8,
                         tied_B = 1,
                         e_step = E_Step_Scan(
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
        self.m = m
        self.D = D
        self.N = N


    def test_expected_log_prob_vhs_batch_match(self):
        """ verifies that expected_log_prob_vhs = mean(expected_log_prob_vhs_batch)
            expected_log_prob_vhs_batch is implemented in terms of expected_energy_vhs
            so this verifies that as well """

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


    def test_d_kl_d_h(self):

        "tests that the gradient of the kl with respect to h matches my analytical version of it "

        model = self.model
        ip = self.model.e_step
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S


        sigma0 = ip.infer_var_s0_hat()
        Sigma1 = ip.infer_var_s1_hat()
        mu0 = T.zeros_like(model.mu)

        trunc_kl = ip.truncated_KL( V = X, obs = { 'H_hat' : H_var,
                                                 'S_hat' : S_var,
                                                 'var_s0_hat' : sigma0,
                                                 'var_s1_hat' : Sigma1 } ).sum()

        assert len(trunc_kl.type.broadcastable) == 0

        grad_H = T.grad(trunc_kl, H_var)

        grad_func = function([H_var, S_var], grad_H)

        grad_theano = grad_func(H,S)


        half = as_floatX(0.5)
        one = as_floatX(1.)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        e = as_floatX(np.e)
        mu = self.model.mu
        alpha = self.model.alpha
        W = self.model.W
        B = self.model.B
        w = self.model.w

        term1 = T.log(H_var)
        term2 = -T.log(one - H_var)
        term3 = - half * T.log( Sigma1 *  two * pi * e )
        term4 = half * T.log(sigma0 *  two * pi * e )
        term5 = - self.model.bias_hid
        term6 = half * ( - sigma0 + Sigma1 + T.sqr(S_var) )
        term7 = - mu * alpha * S_var
        term8 = half * T.sqr(mu) * alpha
        term9 = - T.dot(X * self.model.B, self.model.W) * S_var
        term10 = S_var * T.dot(T.dot(H_var * S_var, W.T * B),W)
        term11 = - w * T.sqr(S_var) * H_var
        term12 = half * (Sigma1 + T.sqr(S_var)) * T.dot(B,T.sqr(W))

        analytical = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12

        grad_analytical = function([H_var, S_var], analytical)(H,S)

        if not np.allclose(grad_theano, grad_analytical):
            print 'grad theano: ',(grad_theano.min(), grad_theano.mean(), grad_theano.max())
            print 'grad analytical: ',(grad_analytical.min(), grad_analytical.mean(), grad_analytical.max())
            ad = np.abs(grad_theano-grad_analytical)
            print 'abs diff: ',(ad.min(),ad.mean(),ad.max())
            assert False

    def test_d_negent_d_h(self):

        "tests that the gradient of the negative entropy with respect to h matches my analytical version of it "

        model = self.model
        ip = self.model.e_step
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S


        sigma0 = ip.infer_var_s0_hat()
        Sigma1 = ip.infer_var_s1_hat()
        mu0 = T.zeros_like(model.mu)

        negent = - self.model.entropy_hs( H_hat =  H_var,
                                                 var_s0_hat =  sigma0,
                                                 var_s1_hat = Sigma1 ).sum()

        assert len(negent.type.broadcastable) == 0

        grad_H = T.grad(negent, H_var)

        grad_func = function([H_var, S_var], grad_H, on_unused_input = 'ignore' )

        grad_theano = grad_func(H,S)


        half = as_floatX(0.5)
        one = as_floatX(1.)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        e = as_floatX(np.e)
        mu = self.model.mu
        alpha = self.model.alpha
        W = self.model.W
        B = self.model.B
        w = self.model.w

        term1 = T.log(H_var)
        term2 = -T.log(one - H_var)
        term3 = - half * T.log( Sigma1 * two * pi * e )
        term4 = half * T.log(  sigma0 * two * pi * e )

        analytical = term1 + term2 + term3 + term4

        grad_analytical = function([H_var, S_var], analytical, on_unused_input = 'ignore')(H,S)

        if not np.allclose(grad_theano, grad_analytical):
            print 'grad theano: ',(grad_theano.min(), grad_theano.mean(), grad_theano.max())
            print 'grad analytical: ',(grad_analytical.min(), grad_analytical.mean(), grad_analytical.max())
            ad = np.abs(grad_theano-grad_analytical)
            print 'abs diff: ',(ad.min(),ad.mean(),ad.max())
            assert False

    def test_d_negent_h_d_h(self):

        "tests that the gradient of the negative entropy of h with respect to \hat{h} matches my analytical version of it "

        model = self.model
        ip = self.model.e_step
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S


        sigma0 = ip.infer_var_s0_hat()
        Sigma1 = ip.infer_var_s1_hat()
        mu0 = T.zeros_like(model.mu)

        negent = - self.model.entropy_h( H_hat =  H_var  ).sum()

        assert len(negent.type.broadcastable) == 0

        grad_H = T.grad(negent, H_var)

        grad_func = function([H_var, S_var], grad_H, on_unused_input = 'ignore')

        grad_theano = grad_func(H,S)


        half = as_floatX(0.5)
        one = as_floatX(1.)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        e = as_floatX(np.e)
        mu = self.model.mu
        alpha = self.model.alpha
        W = self.model.W
        B = self.model.B
        w = self.model.w

        term1 = T.log(H_var)
        term2 = -T.log(one - H_var)

        analytical = term1 + term2

        grad_analytical = function([H_var, S_var], analytical, on_unused_input = 'ignore')(H,S)

        if not np.allclose(grad_theano, grad_analytical):
            print 'grad theano: ',(grad_theano.min(), grad_theano.mean(), grad_theano.max())
            print 'grad analytical: ',(grad_analytical.min(), grad_analytical.mean(), grad_analytical.max())
            ad = np.abs(grad_theano-grad_analytical)
            print 'abs diff: ',(ad.min(),ad.mean(),ad.max())
            assert False


    def test_d_ee_d_h(self):

        "tests that the gradient of the expected energy with respect to h matches my analytical version of it "

        model = self.model
        ip = self.model.e_step
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.001,.999,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S


        sigma0 = ip.infer_var_s0_hat()
        Sigma1 = ip.infer_var_s1_hat()
        mu0 = T.zeros_like(model.mu)

        ee = self.model.expected_energy_vhs( V = X, H_hat = H_var,
                                                 S_hat =  S_var,
                                                 var_s0_hat = sigma0,
                                                 var_s1_hat = Sigma1 ).sum()

        assert len(ee.type.broadcastable) == 0

        grad_H = T.grad(ee, H_var)

        grad_func = function([H_var, S_var], grad_H)

        grad_theano = grad_func(H,S)


        half = as_floatX(0.5)
        one = as_floatX(1.)
        two = as_floatX(2.)
        pi = as_floatX(np.pi)
        e = as_floatX(np.e)
        mu = self.model.mu
        alpha = self.model.alpha
        W = self.model.W
        B = self.model.B
        w = self.model.w

        term1 = - self.model.bias_hid
        term2 = half * ( - sigma0 + Sigma1 + T.sqr(S_var) )
        term3 = - mu * alpha * S_var
        term4 = half * T.sqr(mu) * alpha
        term5 = - T.dot(X * self.model.B, self.model.W) * S_var
        term6 = S_var * T.dot(T.dot(H_var * S_var, W.T * B),W)
        term7 = - w * T.sqr(S_var) * H_var
        term8 = half * (Sigma1 + T.sqr(S_var)) * T.dot(B,T.sqr(W))

        analytical = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

        grad_analytical = function([H_var, S_var], analytical, on_unused_input = 'ignore')(H,S)

        if not np.allclose(grad_theano, grad_analytical):
            print 'grad theano: ',(grad_theano.min(), grad_theano.mean(), grad_theano.max())
            print 'grad analytical: ',(grad_analytical.min(), grad_analytical.mean(), grad_analytical.max())
            ad = np.abs(grad_theano-grad_analytical)
            print 'abs diff: ',(ad.min(),ad.mean(),ad.max())
            assert False

if __name__ == '__main__':
    obj = TestS3C_Misc()
    obj.test_d_ee_d_h()
