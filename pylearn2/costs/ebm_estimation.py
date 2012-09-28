""" Training costs for unsupervised learning of energy-based models """
import theano.tensor as T
from theano import scan
from pylearn2.costs.cost import Cost


class NCE(Cost):
    """ Noise-Contrastive Estimation

        See "Noise-Contrastive Estimation: A new estimation principle for unnormalized models "
        by Gutmann and Hyvarinen

    """
    def h(self, X, model):
        return - T.nnet.sigmoid(self.G(X, model))


    def G(self, X, model):
        return model.log_prob(X) - self.noise.log_prob(X)

    def __call__(self, model, X, Y = None):
        #The Y here is the noise
        #If you don't pass it in, it will be generated internally
        #Passing it in lets you keep it constant while doing
        #a learn search across several theano function calls
        #and stuff like that
        #This interface should probably be changed because it
        #looks too much like the SupervisedCost interface

        if X.name is None:
            X_name = 'X'
        else:
            X_name = X.name


        m_data = X.shape[0]
        m_noise = m_data * self.noise_per_clean

        if Y is None:
            Y = self.noise.random_design_matrix(m_noise)

        #Y = Print('Y',attrs=['min','max'])(Y)

        #hx = self.h(X, model)
        #hy = self.h(Y, model)

        log_hx = -T.nnet.softplus(-self.G(X,model))
        log_one_minus_hy = -T.nnet.softplus(self.G(Y,model))


        #based on equation 3 of the paper
        #ours is the negative of theirs because they maximize it and we minimize it
        rval = -T.mean(log_hx)-T.mean(log_one_minus_hy)

        rval.name = 'NCE('+X_name+')'

        return rval

    def __init__(self, noise, noise_per_clean):
        """
        params
        -------
            noise: a Distribution from which noisy examples are generated
            noise_per_clean: # of noisy examples to generate for each clean example given
        """

        self.noise = noise

        assert isinstance(noise_per_clean, int)
        self.noise_per_clean = noise_per_clean

class SM(Cost):
    """ Score Matching
        See eqn. 4 of "On Autoencoders and Score Matching for Energy Based Models",
        Swersky et al 2011, for details

        Uses the mean over visible units rather than sum over visible units
        so that hyperparameters won't depend as much on the # of visible units
    """
    def __call__(self, model, X):
        X_name = 'X' if X.name is None else X.name

        score = model.score(X)

        sq = 0.5 * T.sqr(score)

        def f(i, fX, fscore):
            score_i_batch = fscore[:,i]
            dummy = score_i_batch.sum()
            full_grad = T.grad(dummy, fX)
            return full_grad[:,i]

        second_derivs, ignored = scan( f, sequences = T.arange(X.shape[1]), non_sequences = [X, score] )
        second_derivs = second_derivs.T

        assert len(second_derivs.type.broadcastable) == 2

        temp = sq + second_derivs

        rval = T.mean(temp)

        rval.name = 'sm('+X_name+')'

        return rval

class SMD(Cost):
    """ Denoising Score Matching
        See eqn. 4.3 of "A Connection Between Score Matching and Denoising Autoencoders"
        by Pascal Vincent for details

        Note that instead of using half the squared norm we use the mean squared error,
        so that hyperparameters don't depend as much on the # of visible units
    """

    def __init__(self, corruptor):
        super(SMD, self).__init__()
        self.corruptor = corruptor

    def __call__(self, model, X):
        X_name = 'X' if X.name is None else X.name

        corrupted_X = self.corruptor(X)

        if corrupted_X.name is None:
            corrupted_X.name = 'corrupt('+X_name+')'
        #

        model_score = model.score(corrupted_X)
        assert len(model_score.type.broadcastable) == len(X.type.broadcastable)
        parzen_score = T.grad( - T.sum(self.corruptor.corruption_free_energy(corrupted_X,X)), corrupted_X)
        assert len(parzen_score.type.broadcastable) == len(X.type.broadcastable)

        score_diff = model_score - parzen_score
        score_diff.name = 'smd_score_diff('+X_name+')'


        assert len(score_diff.type.broadcastable) == len(X.type.broadcastable)


        #TODO: this could probably be faster as a tensordot, but we don't have tensordot for gpu yet
        sq_score_diff = T.sqr(score_diff)

        #sq_score_diff = Print('sq_score_diff',attrs=['mean'])(sq_score_diff)

        smd = T.mean(sq_score_diff)
        smd.name = 'SMD('+X_name+')'

        return smd
