from pylearn2.energy_functions.energy_function import EnergyFunction
import theano.tensor as T

class RBM_EnergyFunction(EnergyFunction):
    def __init__(self):
        pass
    #
#

class GRBM_EnergyFunction(RBM_EnergyFunction):
    def supports_vector_sigma(self):
        raise NotImplementedError()

    def log_P_H_given_V(self, H, V):
        p_one = self.mean_H_given_V(V)

        rval =  T.log(H * p_one + (1.-H) * (1.-p_one)).sum(axis=1)

        return rval

    def mean_H_given_V(self, V):
        raise NotImplementedError()


class GRBM_Type_1(GRBM_EnergyFunction):

    """
        TODO: give a better name

        This GRBM energy function is designed to make
        it easy to interpret score matching as being a denoising autoencoder.

        It is not the same energy function as presented in equation 4.6 of
        Pascal Vincent's paper on denoising autoencoders and score matching,
        because that energy function has no latent variables and therefore
        is not an RBM. However, this is a very similar energy function,
        and has the property that

        J_SMD = (1/sigma)^4 J_DAE

        when the same sigma is used for both the gaussian corruption process
        and the model

        E(v,h) = -(bias_vis^T v + v^T W h + bias_hid^T h - (1/2) v^T v ) / sigma^2
        P(v|h) = N( Wh + bias_vis, 1/sigma^2)
        P(h|v) = sigmoid( (v^T Wh + bias_hid) / sigma^2 )
        F(v) = ( (1/2) v^T v - bias_vis^T v) / sigma^2 - sum_i softplus( ( v^T W + c) / sigma^2 )_i
        score(v) = -( v - bias_vis - sigmoid( (v^T W + bias_hid) / sigma^2 ) W^T )/sigma^2

        This parameterization is motivated by this last property, that the entire score
        function is divided by sigma^2, which makes the equivalence with denosing
        autoencoders possible.

        (As far as I know, I, Ian Goodfellow, just made this parameterization
        of GRBMs up as a way of testing SMD, so don't try to use it to exactly
        reproduce any published GRBM results, as they probably use one of the
        other parameterizations)
    """




    def __init__(self, W, bias_hid, bias_vis, sigma):
        super(GRBM_Type_1,self).__init__()

        self.W = W
        self.bias_hid = bias_hid
        self.bias_vis = bias_vis
        self.sigma = sigma

    @classmethod
    def supports_vector_sigma(cls):
        return False

    def energy(self, varlist):
        V, H = varlist
        return - (
                    T.dot(V, self.bias_vis) +
                    (T.dot(V, self.W) * H).sum(axis=1) +
                    T.dot(H, self.bias_hid) -
                    0.5 * T.sqr(V).sum(axis=1)
                ) / T.sqr(self.sigma)


    def mean_H_given_V(self, V):
        return T.nnet.sigmoid( \
                ( \
                    self.bias_hid + \
                    T.dot(V,self.W) \
                ) / T.sqr(self.sigma) \
                        )
    #

    def reconstruct(self, V):
        H = self.mean_H_given_V(V)
        R = self.mean_V_given_H(H)
        return R
    #

    def mean_V_given_H(self, H):
        return self.bias_vis + T.dot(H,self.W.T)
    #

    def free_energy(self, V):
        V_name = 'V' if V.name is None else V.name

        bias_term = T.dot(V,self.bias_vis)
        bias_term.name = 'bias_term'
        assert len(bias_term.type.broadcastable) == 1

        sq_term = 0.5 * T.sqr(V).sum(axis=1)
        sq_term.name = 'sq_term'
        assert len(sq_term.type.broadcastable) == 1

        softplus_term =  T.nnet.softplus( (T.dot(V,self.W)+self.bias_hid) / T.sqr(self.sigma)).sum(axis=1)
        assert len(softplus_term.type.broadcastable) == 1
        softplus_term.name = 'softplus_term'

        return (
                sq_term
                - bias_term
                ) / T.sqr(self.sigma) - softplus_term
    #

    def score(self, V):
        #score(v) = ( v - bias_vis - sigmoid( beta v^T W + bias_hid ) W^T )/sigma^2

        return -( V \
                - self.reconstruct(V) \
                ) / \
            T.sqr(self.sigma)
    #
#

def grbm_type_1():
    return GRBM_Type_1
