"""
Tests for the show_weights.py script
"""
from theano.compat.six.moves import cPickle
import os

from pylearn2.testing.skip import skip_if_no_matplotlib
from pylearn2.models.mlp import MLP, Linear
from pylearn2.scripts.show_weights import show_weights


def test_show_weights():
    """
    Create a pickled model and show the weights
    """
    skip_if_no_matplotlib()
    with open('model.pkl', 'wb') as f:
        model = MLP(layers=[Linear(dim=1, layer_name='h0', irange=0.1)],
                    nvis=784)
        model.dataset_yaml_src = """
!obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train'
}
"""
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    show_weights('model.pkl', rescale='individual',
                 border=True, out='garbage.png')
    os.remove('model.pkl')
    os.remove('garbage.png')
