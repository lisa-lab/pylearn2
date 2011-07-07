""" Training costs for unsupervised learning of energy-based models """
import theano.tensor as T
from theano import scan
from theano.printing import Print


class NCE:
    """ Noise-Contrastive Estimation

        See "Noise-Contrastive Estimation: A new estimation principle for unnormalized models "
        by Gutmann and Hyvarinen

    """

    def h(self, X, model):
        return - T.nnet.sigmoid(self.G(X, model))


    def G(self, X, model):
        return model.log_prob(X) - self.noise.log_prob(X)

    def __call__(self, model, X):
        try:
            m = X.shape[0]
        except:
            print 'X: '+str(X)
            print 'X.shape: '+str(X.shape)
            print 'X.shape[0]: '+str(X.shape[0])
            assert False

        Y = self.noise.random_design_matrix(m)

        #Y = Print('Y',attrs=['min','max'])(Y)

        #hx = self.h(X, model)
        #hy = self.h(Y, model)

        log_hx = T.nnet.softplus(-self.G(X,model))
        log_one_minus_hy = T.nnet.softplus(self.G(Y,model))


        rval = T.mean(log_hx+log_one_minus_hy)

        #rval = Print('nce cost',attrs=['min','max'])(rval)

        return rval

    def __init__(self, noise):
        self.noise = noise

class SM:
    """ Score Matching
        See eqn. 4 of "On Autoencoders and Score Matching for Energy Based Models",
        Swersky et al 2011, for details
    """

    def __init__(self):
        pass
    #

    def __call__(self, model, X):
        score = model.score(X)

        sq = 0.5 * T.sqr(score)

        def f(i, fX, fscore):
            score_i_batch = fscore[:,i]
            dummy = score_i_batch.sum()
            full_grad = T.grad(dummy, fX)
            return full_grad[:,i]
        #

        second_derivs, ignored = scan( f, sequences = T.arange(X.shape[1]), non_sequences = [X, score] )
        second_derivs = second_derivs.T

        assert len(second_derivs.type.broadcastable) == 2

        temp = sq + second_derivs

        rval = T.mean(temp)

        return rval

class SMD:
    """ Denoising Score Matching
        See eqn. 5 of "On Autoencoders and Score Matching for Energy Based Models",
        Swersky et al 2011, for details
    """

    def __init__(self, corruptor):
        self.corruptor = corruptor
    #

    def __call__(self, model, X):
        X_name = 'X' if X.name is None else X.name

        corrupted_X = self.corruptor(X)

        if corrupted_X.name is None:
            corrupted_X.name = 'corrupt('+X_name+')'
        #

        clean_score = model.score(X)
        dirty_score = model.score(corrupted_X)

        score_diff = dirty_score - clean_score
        score_diff.name = 'smd_score_diff('+X_name+')'

        #TODO: this could probably be faster as a tensordot, but we don't have tensordot for gpu yet
        smd = T.mean(T.sqr(score_diff))
        smd.name = 'SMD('+X_name+')'

        return smd
    #
#
