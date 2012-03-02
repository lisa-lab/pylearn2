from pylearn2.models.rbm import RBM

def test_get_weights():

    model = RBM(nvis = 2, nhid = 3)

    W = model.get_weights()

def test_get_input_space():

    model = RBM(nvis = 2, nhid = 3)

    space = model.get_input_space()
