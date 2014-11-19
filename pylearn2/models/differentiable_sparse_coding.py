"""
An implementation of the model described in "Differentiable Sparse Coding" by
Bradley and Bagnell
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import numpy as N
from theano.compat.six.moves import xrange
import theano.tensor as T

import theano
from theano import function, shared, config
floatX = config.floatX
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


class DifferentiableSparseCoding(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    nvis : WRITEME
    nhid : WRITEME
    init_lambda : WRITEME
    init_p : WRITEME
    init_alpha : WRITEME
    learning_rate : WRITEME
    """

    def __init__(self, nvis, nhid,
            init_lambda,
            init_p, init_alpha, learning_rate):
        self.nvis = int(nvis)
        self.nhid = int(nhid)
        self.init_lambda = float(init_lambda)
        self.init_p = float(init_p)
        self.init_alpha = N.cast[config.floatX](init_alpha)
        self.tol = 1e-6
        self.time_constant = 1e-2
        self.learning_rate = N.cast[config.floatX](learning_rate)

        self.predictor_learning_rate = self.learning_rate

        self.rng = make_np_rng(None, [1,2,3], which_method="randn")

        self.error_record = []
        self.ERROR_RECORD_MODE_MONITORING = 0
        self.error_record_mode = self.ERROR_RECORD_MODE_MONITORING

        self.instrumented = False

        self.redo_everything()

    def get_output_dim(self):
        """
        .. todo::

            WRITEME
        """
        return self.nhid

    def get_output_channels(self):
        """
        .. todo::

            WRITEME
        """
        return self.nhid

    def normalize_W(self):
        """
        .. todo::

            WRITEME
        """
        W = self.W.get_value(borrow=True)
        norms = N.sqrt(N.square(W).sum(axis=0))
        self.W.set_value(W/norms, borrow=True)

    def redo_everything(self):
        """
        .. todo::

            WRITEME
        """
        self.W = shared(N.cast[floatX](self.rng.randn(self.nvis,self.nhid)), name='W')

        self.pred_W = shared(self.W.get_value(borrow=False),name='pred_W')
        self.pred_b = shared(N.zeros(self.nhid,dtype=floatX),name='pred_b')
        self.pred_g = shared(N.ones(self.nhid,dtype=floatX),name='pred_g')


        self.normalize_W()
        self.p = shared(N.zeros(self.nhid, dtype=floatX)+N.cast[floatX](self.init_p), name='p')

        #mispelling lambda because python is too dumb to know that self.lambda isn't a lambda function
        self.lamda = shared(
                            N.zeros(
                                    self.nhid, dtype=floatX)+
                            N.cast[floatX](self.init_lambda), name='lambda')


        self.alpha = self.init_alpha
        self.failure_rate = .5

        self.examples_seen = 0
        self.batches_seen = 0

        self.redo_theano()

    def recons_error(self, v, h):
        """
        .. todo::

            WRITEME
        """
        recons = T.dot(self.W,h)
        diffs = recons - v
        rval = T.dot(diffs,diffs) / N.cast[floatX](self.nvis)
        return rval

    def recons_error_batch(self, V, H):
        """
        .. todo::

            WRITEME
        """
        recons = T.dot(H,self.W.T)
        diffs = recons - V
        rval = T.mean(T.sqr(diffs))
        return rval

    def sparsity_penalty(self, v, h):
        """
        .. todo::

            WRITEME
        """
        sparsity_measure = h * T.log(h) - h * T.log(self.p) - h + self.p
        rval = T.dot(self.lamda, sparsity_measure) / N.cast[floatX](self.nhid)
        return rval

    def sparsity_penalty_batch(self, V, H):
        """
        .. todo::

            WRITEME
        """
        sparsity_measure = H * T.log(H) - H * T.log(self.p) - H + self.p
        sparsity_measure_exp = T.mean(sparsity_measure, axis=0)
        rval = T.dot(self.lamda, sparsity_measure_exp) / N.cast[floatX](self.nhid)
        return rval


    def coding_obj(self, v, h):
        """
        .. todo::

            WRITEME
        """
        return self.recons_error(v,h) + self.sparsity_penalty(v,h)

    def coding_obj_batch(self, V, H):
        """
        .. todo::

            WRITEME
        """
        return self.recons_error_batch(V,H) + self.sparsity_penalty_batch(V,H)

    def predict(self, V):
        """
        .. todo::

            WRITEME
        """
        rval =  T.nnet.sigmoid(T.dot(V,self.pred_W)+self.pred_b)*self.pred_g
        assert rval.type.dtype == V.type.dtype
        return rval

    def redo_theano(self):
        """
        .. todo::

            WRITEME
        """

        self.h = shared(N.zeros(self.nhid, dtype=floatX), name='h')
        self.v = shared(N.zeros(self.nvis, dtype=floatX), name='v')

        input_v = T.vector()
        assert input_v.type.dtype == floatX

        self.init_h_v = function([input_v], updates = { self.h : self.predict(input_v),
                                                 self.v : input_v } )


        coding_obj = self.coding_obj(self.v, self.h)
        assert len(coding_obj.type.broadcastable) == 0

        coding_grad = T.grad(coding_obj, self.h)
        assert len(coding_grad.type.broadcastable) == 1

        self.coding_obj_grad = function([], [coding_obj, coding_grad] )


        self.new_h = shared(N.zeros(self.nhid, dtype=floatX), name='new_h')

        alpha = T.scalar(name='alpha')

        outside_grad = T.vector(name='outside_grad')

        new_h = T.clip(self.h * T.exp(-alpha * outside_grad), 1e-10, 1e4)

        new_obj = self.coding_obj(self.v, new_h)

        self.try_step = function( [alpha, outside_grad], updates = { self.new_h : new_h }, outputs = new_obj )

        self.accept_h = function( [], updates = { self.h : self.new_h } )

        self.get_h = function( [] , self.h )


        V = T.matrix(name='V')
        H = T.matrix(name='H')

        coding_obj_batch = self.coding_obj_batch(V,H)

        self.code_learning_obj = function( [V,H], coding_obj_batch)

        learning_grad = T.grad( coding_obj_batch, self.W )
        self.code_learning_step = function( [V,H,alpha], updates = { self.W : self.W - alpha * learning_grad } )



        pred_obj = T.mean(T.sqr(self.predict(V)-H))

        predictor_params = [ self.pred_W, self.pred_b, self.pred_g ]

        pred_grads = T.grad(pred_obj, wrt = predictor_params )

        predictor_updates = {}

        for param, grad in zip(predictor_params, pred_grads):
            predictor_updates[param] = param - alpha * grad

        predictor_updates[self.pred_g ] = T.clip(predictor_updates[self.pred_g], N.cast[floatX](0.5), N.cast[floatX](1000.))

        self.train_predictor = function([V,H,alpha] , updates = predictor_updates )

    def weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ['v','h']

    def error_func(self, x):
        """
        .. todo::

            WRITEME
        """
        batch_size = x.shape[0]

        H = N.zeros((batch_size,self.nhid),dtype=floatX)

        for i in xrange(batch_size):
            assert self.alpha > 9e-8
            H[i,:] = self.optimize_h(x[i,:])
            assert self.alpha > 9e-8

        return self.code_learning_obj(x,H)

    def record_monitoring_error(self, dataset, batch_size, batches):
        """
        .. todo::

            WRITEME
        """
        logger.info('running on monitoring set')
        assert self.error_record_mode == self.ERROR_RECORD_MODE_MONITORING

        w = self.W.get_value(borrow=True)
        logger.info('weights summary: '
                    '({0}, {1}, {2})'.format(w.min(), w.mean(), w.max()))

        errors = []

        if self.instrumented:
            self.clear_instruments()

        for i in xrange(batches):
            x = dataset.get_batch_design(batch_size)
            error = self.error_func(x)
            errors.append( error )
            if self.instrumented:
                self.update_instruments(x)


        self.error_record.append( (self.examples_seen, self.batches_seen, N.asarray(errors).mean() ) )


        if self.instrumented:
            self.instrument_record.begin_report(examples_seen = self.examples_seen, batches_seen = self.batches_seen)
            self.make_instrument_report()
            self.instrument_record.end_report()
            self.clear_instruments()
        logger.info('monitoring set done')

    def infer_h(self, v):
        """
        .. todo::

            WRITEME
        """
        return self.optimize_h(v)

    def optimize_h(self, v):
        """
        .. todo::

            WRITEME
        """
        assert self.alpha > 9e-8

        self.init_h_v(v)

        first = True

        while True:
            obj, grad = self.coding_obj_grad()

            if first:
                #print 'orig_obj: ', obj
                first = False

            assert not N.any(N.isnan(obj))
            assert not N.any(N.isnan(grad))

            if N.abs(grad).max() < self.tol:
                break

            #print 'max gradient ',N.abs(grad).max()

            cur_alpha = N.cast[floatX] ( self.alpha  + 0.0 )

            new_obj = self.try_step(cur_alpha, grad)

            assert not N.isnan(new_obj)

            self.failure_rate =  (1. - self.time_constant ) * self.failure_rate + self.time_constant * float(new_obj > obj )

            assert self.alpha > 9e-8

            if self.failure_rate  > .6 and self.alpha > 1e-7:
                self.alpha *= .9
                #print '!!!!!!!!!!!!!!!!!!!!!!shrank alpha to ',self.alpha
            elif self.failure_rate < .3:
                self.alpha *= 1.1
                #print '**********************grew alpha to ',self.alpha

            assert self.alpha > 9e-8

            while new_obj >= obj:
                cur_alpha *= .9

                if cur_alpha < 1e-12:
                    self.accept_h()
                    #print 'failing final obj ',new_obj
                    return self.get_h()

                new_obj = self.try_step(cur_alpha, grad)

                assert not N.isnan(new_obj)

            self.accept_h()

        #print 'final obj ',new_obj
        return self.get_h()

    def train_batch(self, dataset, batch_size):
        """
        .. todo::

            WRITEME
        """
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
        return True

    def learn_mini_batch(self, x):
        """
        .. todo::

            WRITEME
        """
        assert self.alpha > 9e-8

        batch_size = x.shape[0]

        H = N.zeros((batch_size,self.nhid),dtype=floatX)

        for i in xrange(batch_size):
            assert self.alpha > 9e-8
            H[i,:] = self.optimize_h(x[i,:])
            assert self.alpha > 9e-8

        self.code_learning_step(x,H,self.learning_rate)
        self.normalize_W()

        self.train_predictor(x,H,self.predictor_learning_rate)

        self.examples_seen += x.shape[0]
        self.batches_seen += 1
