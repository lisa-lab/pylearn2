"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
from pylearn2.energy_functions.energy_function import EnergyFunction
import theano.tensor as T

class RBM_EnergyFunction(EnergyFunction):
    """
    .. todo::

        WRITEME
    """

    def __init__(self):
        pass

class GRBM_EnergyFunction(RBM_EnergyFunction):
    """
    .. todo::

        WRITEME
    """

    def supports_vector_sigma(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    def log_P_H_given_V(self, H, V):
        """
        .. todo::

            WRITEME
        """
        p_one = self.mean_H_given_V(V)

        rval =  T.log(H * p_one + (1.-H) * (1.-p_one)).sum(axis=1)

        return rval

    def mean_H_given_V(self, V):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()


class GRBM_Type_1(GRBM_EnergyFunction):
    """
    .. todo::

        WRITEME properly
    
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
    P(v|h) = N( Wh + bias_vis, sigma^2)
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

    Parameters
    ----------
    transformer : WRITEME
    bias_hid : WRITEME
    bias_vis : WRITEME
    sigma : WRITEME
    """

    def __init__(self, transformer, bias_hid, bias_vis, sigma):
        super(GRBM_Type_1,self).__init__()

        self.transformer = transformer
        self.bias_hid = bias_hid
        self.bias_vis = bias_vis
        self.sigma = sigma

    @classmethod
    def supports_vector_sigma(cls):
        """
        .. todo::

            WRITEME
        """
        return False

    def energy(self, varlist):
        """
        .. todo::

            WRITEME
        """
        V, H = varlist
        return - (
                    T.dot(V, self.bias_vis) +
                    (self.transformer.lmul(V) * H).sum(axis=1) +
                    T.dot(H, self.bias_hid) -
                    0.5 * T.sqr(V).sum(axis=1)
                ) / T.sqr(self.sigma)


    def mean_H_given_V(self, V):
        """
        .. todo::

            WRITEME
        """
        V_name = 'V'
        if hasattr(V, 'name') and V.name is not None:
            V_name = V.name

        rval =  T.nnet.sigmoid( \
                ( \
                    self.bias_hid + \
                    self.transformer.lmul(V) \
                ) / T.sqr(self.sigma) \
                        )

        rval.name = 'mean_H_given_V( %s )' % V_name

        return rval

    def reconstruct(self, V):
        """
        .. todo::

            WRITEME
        """
        H = self.mean_H_given_V(V)
        R = self.mean_V_given_H(H)
        return R

    def mean_V_given_H(self, H):
        """
        .. todo::

            WRITEME
        """
        H_name = 'H'
        if hasattr(H,'name') and H.name is not None:
            H_name = H.name

        transpose = self.transformer.lmul_T(H)
        transpose.name = 'transpose'

        rval =  self.bias_vis + transpose
        rval.name = 'mean_V_given_H(%s)' % H_name

        return rval

    def free_energy(self, V):
        """
        .. todo::

            WRITEME
        """
        V_name = 'V' if V.name is None else V.name

        assert V.ndim == 2

        bias_term = T.dot(V,self.bias_vis)
        bias_term.name = 'bias_term'
        assert len(bias_term.type.broadcastable) == 1

        sq_term = 0.5 * T.sqr(V).sum(axis=1)
        sq_term.name = 'sq_term'
        assert len(sq_term.type.broadcastable) == 1

        softplus_term =  T.nnet.softplus( (self.transformer.lmul(V)+self.bias_hid) / T.sqr(self.sigma)).sum(axis=1)
        assert len(softplus_term.type.broadcastable) == 1
        softplus_term.name = 'softplus_term'

        return (
                sq_term
                - bias_term
                ) / T.sqr(self.sigma) - softplus_term

    def score(self, V):
        """
        .. todo::

            WRITEME
        """
        #score(v) = ( v - bias_vis - sigmoid( beta v^T W + bias_hid ) W^T )/sigma^2

        rval =  -( V \
                - self.reconstruct(V) \
                ) / \
            T.sqr(self.sigma)

        rval.name = 'score'

        return rval

def grbm_type_1():
    """
    .. todo::

        WRITEME
    """
    return GRBM_Type_1
