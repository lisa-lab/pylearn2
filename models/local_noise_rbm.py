import numpy as N
from theano import function, scan, shared
import theano.tensor as T
import copy
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
import theano
floatX = theano.config.floatX


class LocalNoiseRBM(object):
    def reset_rng(self):

        self.rng = N.random.RandomState([12.,9.,2.])
        self.theano_rng = RandomStreams(self.rng.randint(2**30))
        if self.initialized:
            self.redo_theano()
    #

    def __getstate__(self):
        d = copy.copy(self.__dict__)

        #remove everything set up by redo_theano

        for name in self.names_to_del:
            if name in d:
                del d[name]

        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        #self.redo_theano()      # todo: make some way of not running this, so it's possible to just open something up and look at its weights fast without recompiling it

    def weights_format(self):
        return ['v','h']

    def get_dimensionality(self):
        return 0

    def important_error(self):
        return 2

    def __init__(self, nvis, nhid,
                learning_rate, irange,
                init_bias_hid,
                init_beta,
                min_misclass,
                max_misclass,
                time_constant,
                beta_scale_up,
                beta_scale_down,
                max_beta
                ):
        self.initialized = False
        self.reset_rng()
        self.nhid = nhid
        self.nvis = nvis
        self.learning_rate = learning_rate
        self.ERROR_RECORD_MODE_MONITORING = 0
        self.error_record_mode = self.ERROR_RECORD_MODE_MONITORING
        self.init_weight_mag = irange
        self.force_batch_size = 0
        self.init_bias_hid = init_bias_hid
        self.beta = shared(N.cast[floatX] (init_beta))
        self.min_misclass = min_misclass
        self.max_misclass = max_misclass
        self.time_constant = time_constant
        self.beta_scale_up = beta_scale_up
        self.beta_scale_down = beta_scale_down
        self.max_beta = max_beta
        self.misclass = -1

        self.names_to_del = []

        self.redo_everything()

    def set_error_record_mode(self, mode):
        self.error_record_mode = mode

    def set_size_from_dataset(self, dataset):
        self.nvis = dataset.get_output_dim()
        self.redo_everything()
        self.vis_mean.set_value( dataset.get_marginals(), borrow=False)
    #

    def get_input_dim(self):
        return self.nvis

    def get_output_dim(self):
        return self.nhid

    def redo_everything(self):
        self.initialized = True

        self.error_record = []
        self.examples_seen = 0
        self.batches_seen = 0

        self.W = shared( N.cast[floatX](self.rng.uniform(-self.init_weight_mag, self.init_weight_mag, (self.nvis, self.nhid ) ) ))
        self.W.name = 'W'

        self.b = shared( N.cast[floatX](N.zeros(self.nhid) + self.init_bias_hid) )
        self.b.name = 'b'

        self.c = shared( N.cast[floatX](N.zeros(self.nvis)))
        self.c.name = 'c'

        self.params = [ self.W, self.c, self.b ]

        self.redo_theano()
    #


    def batch_energy(self, V, H):

        output_scan, updates = scan(
                 lambda v, h: 0.5 * T.dot(v,v) - T.dot(self.b,h) - T.dot(self.c,v) -T.dot(v,T.dot(self.W,h)),
                 sequences  = (V,H))


        return output_scan

    def p_h_given_v(self, V):
        return T.nnet.sigmoid(self.b + T.dot(V,self.W))

    def batch_free_energy(self, V):

        output_scan, updates = scan(
                lambda v: 0.5 * T.dot(v,v) - T.dot(self.c,v) - T.sum(T.nnet.softplus( T.dot(v,self.W)+self.b)),
                 sequences  = V)


        return output_scan

    def redo_theano(self):

        pre_existing_names = dir(self)

        self.W_T = self.W.T
        self.W_T.name = 'W.T'

        alpha = T.scalar()

        X = T.matrix()
        X.name = 'X'


        corrupted = self.theano_rng.normal(size = X.shape, avg = X,
                                    std = T.sqrt(self.beta), dtype = X.dtype)


        corrupted = corrupted * ( T.sqr(X).sum(axis=1) / T.sqr(corrupted).sum(axis=1)).dimshuffle(0,'x')

        self.corruption_func = function([X],corrupted)

        E_c = self.batch_free_energy(corrupted)
        E_d = self.batch_free_energy(X)

        obj = T.mean(
                -T.log(
                    T.nnet.sigmoid(
                        E_c - E_d)   ) )

        self.error_func = function([X],obj )

        #print 'maker'
        #print theano.printing.debugprint(self.error_func.maker.env.outputs[0])
        #print 'obj'
        #print theano.printing.debugprint(obj)

        self.misclass_func = function([X], (E_c < E_d ).mean())
        #self.misclass_func = function([X], ( T.sum(T.sqr(corrupted),axis=1) < T.sum(T.sqr(X),axis=1) ).mean())
        #self.misclass_func = function([X], T.sum(T.sqr(corrupted),axis=1).mean())
        #self.misclass_func = function([X], T.sum(T.sqr(X),axis=1).mean())

        grads = [ T.grad(obj,param) for param in self.params ]

        self.learn_func = function([X, alpha], updates =
                [ (param, param - alpha * grad) for (param,grad)
                    in zip(self.params, grads) ] , name='learn_func')

        self.recons_func = function([X], self.gibbs_step_exp(X) , name = 'recons_func')

        post_existing_names = dir(self)

        self.names_to_del = [ name for name in post_existing_names if name not in pre_existing_names]



    def learn(self, dataset, batch_size):
        self.learn_mini_batch(dataset.get_batch_design(batch_size))


    def recons_func(self, x):
        rval = N.zeros(x.shape)
        for i in xrange(x.shape[0]):
            rval[i,:] = self.gibbs_step_exp(x[i,:])

        return rval

    def record_monitoring_error(self, dataset, batch_size, batches):
        assert self.error_record_mode == self.ERROR_RECORD_MODE_MONITORING

        print 'beta (variance): '+str(self.beta.get_value())

        #always use the same seed for monitoring error
        self.theano_rng.seed(5)

        errors = []

        misclasses = []

        for i in xrange(batches):
            x = dataset.get_batch_design(batch_size)
            assert x.shape == (batch_size, self.nvis)
            error = self.error_func(x)
            errors.append( error )
            misclass = self.misclass_func(x)
            misclasses.append(misclass)
        #

        misclass = N.asarray(misclasses).mean()

        print 'misclassification rate: '+str(misclass)

        self.error_record.append( (self.examples_seen, self.batches_seen, N.asarray(errors).mean(), self.beta.get_value(), misclass ) )

        print "TODO: restore old theano_rng state instead of jumping to new one"
        self.theano_rng.seed(self.rng.randint(2**30))
    #

    def reconstruct(self, x, use_noise):
        assert x.shape[0] == 1

        print 'x summary: '+str((x.min(),x.mean(),x.max()))

        #this method is mostly a hack to make the formatting work the same as denoising autoencoder
        self.truth_shared = shared(x.copy())

        if use_noise:
            self.vis_shared = shared(self.corruption_func(x))
        else:
            self.vis_shared = shared(x.copy())

        self.reconstruction = self.recons_func(self.vis_shared.get_value())

        print 'recons summary: '+str((self.reconstruction.min(),self.reconstruction.mean(),self.reconstruction.max()))


    def gibbs_step_exp(self, V):
        base_name = V.name

        if base_name is None:
            base_name = 'anon'

        Q = self.p_h_given_v(V)
        H = self.sample_hid(Q)

        H.name =  base_name + '->hid_sample'

        sample =  self.c + T.dot(H,self.W_T)

        sample.name = base_name + '->sample_expectation'

        return sample


    def sample_hid(self, Q):
        return self.theano_rng.binomial(size = Q.shape, n = 1, p = Q,
                                dtype = Q.dtype)


    def learn_mini_batch(self, x):

        assert x.shape[1] == self.nvis

        cur_misclass = self.misclass_func(x)

        if self.misclass == -1:
            self.misclass = cur_misclass
        else:
            self.misclass = self.time_constant * cur_misclass + (1.-self.time_constant) * self.misclass

        #print 'current misclassification rate: '+str(self.misclass)

        if self.misclass > self.max_misclass:
            self.beta.set_value(min(self.max_beta,self.beta.get_value() * self.beta_scale_up) )
        elif self.misclass < self.min_misclass:
            self.beta.set_value(self.beta.get_value() * self.beta_scale_down )
        #

        #print 'new beta: '+str(self.beta.get_value())

        self.learn_func(x, self.learning_rate)



        self.examples_seen += x.shape[0]
        self.batches_seen += 1
    #
#

