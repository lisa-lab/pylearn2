import numpy as np

import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from theano import tensor as T

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.rbm import RBM
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.utils.rng import make_theano_rng


def test_get_weights():
    # Tests that the RBM, when constructed
    # with nvis and nhid arguments, supports the
    # weights interface

    model = RBM(nvis=2, nhid=3)
    W = model.get_weights()


def test_get_input_space():
    # Tests that the RBM supports
    # the Space interface

    model = RBM(nvis=2, nhid=3)
    space = model.get_input_space()


def test_gibbs_step_for_v():
    # Just tests that gibbs_step_for_v can be called
    # without crashing

    model = RBM(nvis=2, nhid=3)

    theano_rng = make_theano_rng(17, which_method='binomial')

    X = T.matrix()

    Y = model.gibbs_step_for_v(X, theano_rng)


def test_train_batch():
    # Just tests that train_batch can be called without crashing

    m = 1
    dim = 2
    rng = np.random.RandomState([2014, 3, 17])
    X = rng.randn(m, dim)
    train = DenseDesignMatrix(X=X)

    rbm = RBM(nvis=dim, nhid=3)
    trainer = DefaultTrainingAlgorithm(batch_size=1, batches_per_iter=10)
    trainer.setup(rbm, train)
    trainer.train(train)
