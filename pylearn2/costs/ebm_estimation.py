""" Training costs for unsupervised learning of energy-based models """
import theano.tensor as T
from theano import scan
from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace
from pylearn2.utils import py_integer_types


class NCE(Cost):
    """ Noise-Contrastive Estimation

        See "Noise-Contrastive Estimation: A new estimation principle for unnormalized models "
        by Gutmann and Hyvarinen

    """
    def h(self, X, model):
        return - T.nnet.sigmoid(self.G(X, model))


    def G(self, X, model):
        return model.log_prob(X) - self.noise.log_prob(X)

    def expr(self, model, data, noisy_data=None):
        # noisy_data is not considered part of the data.
        #If you don't pass it in, it will be generated internally
        #Passing it in lets you keep it constant while doing
        #a learn search across several theano function calls
        #and stuff like that
        space, source = self.get_data_specs(model)
        space.validate(data)
        X = data
        if X.name is None:
            X_name = 'X'
        else:
            X_name = X.name

        m_data = X.shape[0]
        m_noise = m_data * self.noise_per_clean

        if noisy_data is not None:
            space.validate(noisy_data)
            Y = noisy_data
        else:
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

        assert isinstance(noise_per_clean, py_integer_types)
        self.noise_per_clean = noise_per_clean

    def get_data_specs(self, model):
        space = model.get_input_space()
        source = model.get_input_source()
        return (space, source)


class SM(Cost):
    """ (Regularized) Score Matching
        
        See:
        - "Regularized estimation of image statistics by Score Matching",
          D. Kingma, Y. LeCun, NIPS 2010
        - eqn. 4 of "On Autoencoders and Score Matching for Energy Based Models"
          Swersky et al 2011
        
        Uses the mean over visible units rather than sum over visible units
        so that hyperparameters won't depend as much on the # of visible units
    """
    
    def __init__(self, lambd = 0):
        assert lambd >= 0
        self.lambd = lambd

    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        X = data
        X_name = 'X' if X.name is None else X.name

        def f(i, _X, _dx):
            return T.grad(_dx[:,i].sum(), _X)[:,i]

        dx = model.score(X)
        ddx, _ = scan(f, sequences = [T.arange(X.shape[1])], non_sequences = [X, dx])
        ddx = ddx.T

        assert len(ddx.type.broadcastable) == 2

        rval = T.mean(0.5 * dx**2 + ddx + self.lambd * ddx**2)
        rval.name = 'sm('+X_name+')'

        return rval

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())


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

    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        X = data
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

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())
