"""
A unit test for the summarize_model.py script
"""
from theano.compat.six.moves import cPickle
import os

from pylearn2.testing.skip import skip_if_no_matplotlib
from pylearn2.models.mlp import MLP, Linear
from pylearn2.scripts.summarize_model import summarize


def test_summarize_model():
    """
    Asks the summarize_model.py script to inspect a pickled model and
    check that it completes succesfully
    """
    skip_if_no_matplotlib()
    with open('model.pkl', 'wb') as f:
        cPickle.dump(MLP(layers=[Linear(dim=5, layer_name='h0', irange=0.1)],
                         nvis=10), f, protocol=cPickle.HIGHEST_PROTOCOL)
    summarize('model.pkl')
    os.remove('model.pkl')
