from pylearn2.models.model import Model
from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T

class DiagonalMND(Model):
    """A model based on the multivariate normal distribution
    This variant is constrained to have diagonal covariance
    TODO: unify this with distribution.mnd"""

    def __init__(self, nvis,
            init_beta,
            init_mu,
            min_beta,
            max_beta):

        #copy all arguments to the object
        self.__dict__.update( locals() )
        del self.self

        super(DiagonalMND,self).__init__()

        #build the object
        self.redo_everything()

    def redo_everything(self):

        self.beta = sharedX(np.ones((self.nvis,))*self.init_beta,'beta')
        self.mu = sharedX(np.ones((self.nvis,))*self.init_mu,'mu')
        self.redo_theano()


    def free_energy(self, X):

        diff = X-self.mu
        sq = T.sqr(diff)

        return  0.5 * T.dot( sq, self.beta )

    def redo_theano(self):

        init_names = dir(self)

        self.censored_updates = {}
        for param in self.get_params():
            self.censored_updates[param] = set([])

        final_names = dir(self)

        self.register_names_to_del( [name for name in final_names if name not in init_names])


    def censor_updates(self, updates):

        if self.beta in updates and updates[self.beta] not in self.censored_updates[self.beta]:
            updates[self.beta] = T.clip(updates[self.beta], self.min_beta, self.max_beta )

        params = self.get_params()
        for param in updates:
            if param in params:
                self.censored_updates[param] = self.censored_updates[param].union(set([updates[param]]))

    def get_params(self):
        return [self.mu, self.beta ]


def kl_divergence(q,p):
    #KL divergence of two DiagonalMNDs
    #http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#KL_divergence_for_Normal_Distributions
    #D_KL(q||p) = 0.5 ( beta_q^T beta_p^-1 + beta_p^T sq(mu_p - mu_q) - log(det Siga_q / det Sigma_p) - k)
    assert isinstance(q,DiagonalMND)
    assert isinstance(p,DiagonalMND)

    assert q.nvis == p.nvis
    k = q.nvis

    beta_q = q.beta
    beta_p = p.beta
    beta_p_inv = 1./beta_p

    trace_term = T.dot(beta_q,beta_p_inv)

    mu_p = p.mu
    mu_q = q.mu

    quad_term = T.dot(beta_p, T.sqr(mu_p-mu_q))

    # - log ( det Sigma_q / det Sigma_p)
    # = log det Sigma_p - log det Sigma_q
    # = log det Beta_p_inv - log det Beta_q_inv
    # = sum(log(beta_p_inv)) - sum(log(beta_q_inv))
    # = sum(log(beta_q)) - sum(log(beta_p))
    log_term = T.sum(T.log(beta_q)) - T.sum(T.log(beta_p))

    inside_parens = trace_term + quad_term + log_term - k

    rval = 0.5 * inside_parens
