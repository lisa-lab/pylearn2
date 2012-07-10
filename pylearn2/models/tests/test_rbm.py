from pylearn2.models.rbm import RBM
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from theano import tensor as T

def test_get_weights():
    #Tests that the RBM, when constructed
    #with nvis and nhid arguments, supports the
    #weights interface

    model = RBM(nvis = 2, nhid = 3)
    W = model.get_weights()

def test_get_input_space():
    #Tests that the RBM supports
    #the Space interface

    model = RBM(nvis = 2, nhid = 3)
    space = model.get_input_space()

def test_gibbs_step_for_v():
    #Just tests that gibbs_step_for_v can be called
    #without crashing (protection against refactoring
    #damage, aren't interpreted languages great?)

    model = RBM(nvis = 2, nhid = 3)

    theano_rng = RandomStreams(17)

    X = T.matrix()

    Y = model.gibbs_step_for_v(X, theano_rng)
