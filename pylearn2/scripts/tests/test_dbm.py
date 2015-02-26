"""
Tests scripts in the DBM folder
"""

import os
import pylearn2.scripts.dbm.show_negative_chains as negative_chains
import pylearn2.scripts.dbm.show_reconstructions as show_reconstruct
import pylearn2.scripts.dbm.show_samples as show_samples
import pylearn2.scripts.dbm.top_filters as top_filters
from pylearn2.config import yaml_parse
from pylearn2.models.dbm.layer import BinaryVector, BinaryVectorMaxPool
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.dbm.dbm import DBM
from nose.tools import with_setup
from pylearn2.datasets import control
from pylearn2.utils import serial
from theano import function
from theano.compat.six.moves import cPickle


def setup():
    """
    Create pickle file with a simple model.
    """
    # tearDown is guaranteed to run pop_load_data.
    control.push_load_data(False)
    with open('dbm.pkl', 'wb') as f:
        dataset = MNIST(which_set='train', start=0, stop=100, binarize=True)
        vis_layer = BinaryVector(nvis=784, bias_from_marginals=dataset)
        hid_layer1 = BinaryVectorMaxPool(layer_name='h1', pool_size=1,
                                         irange=.05, init_bias=-2.,
                                         detector_layer_dim=50)
        hid_layer2 = BinaryVectorMaxPool(layer_name='h2', pool_size=1,
                                         irange=.05, init_bias=-2.,
                                         detector_layer_dim=10)
        model = DBM(batch_size=20, niter=2, visible_layer=vis_layer,
                    hidden_layers=[hid_layer1, hid_layer2])
        model.dataset_yaml_src = """
!obj:pylearn2.datasets.binarizer.Binarizer {
    raw: !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        start: 0,
        stop: 100
    }
}
"""
        model.layer_to_chains = model.make_layer_to_state(1)
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)


def teardown():
    """Delete the pickle file created for the tests"""
    if os.path.isfile('dbm.pkl'):
        os.remove('dbm.pkl')
    control.pop_load_data()


@with_setup(setup, teardown)
def test_show_negative_chains():
    """Test the show_negative_chains script main function"""
    negative_chains.show_negative_chains('dbm.pkl')


@with_setup(setup, teardown)
def test_show_reconstructions():
    """Test the reconstruction update_viewer function"""
    rows = 5
    cols = 10
    m = rows * cols

    model = show_reconstruct.load_model('dbm.pkl', m)
    dataset = show_reconstruct.load_dataset(model.dataset_yaml_src,
                                            use_test_set='n')

    batch = model.visible_layer.space.make_theano_batch()
    reconstruction = model.reconstruct(batch)
    recons_func = function([batch], reconstruction)

    vis_batch = dataset.get_batch_topo(m)
    patch_viewer = show_reconstruct.init_viewer(dataset, rows, cols, vis_batch)
    show_reconstruct.update_viewer(dataset, batch, rows, cols, patch_viewer,
                                   recons_func, vis_batch)


@with_setup(setup, teardown)
def test_show_samples():
    """Test the samples update_viewer function"""
    rows = 10
    cols = 10
    m = rows * cols

    model = show_samples.load_model('dbm.pkl', m)
    dataset = yaml_parse.load(model.dataset_yaml_src)

    samples_viewer = show_samples.init_viewer(dataset, rows, cols)

    vis_batch = dataset.get_batch_topo(m)
    show_samples.update_viewer(dataset, samples_viewer, vis_batch, rows, cols)


@with_setup(setup, teardown)
def test_top_filters():
    """Test the top_filters viewer functions"""
    model = serial.load('dbm.pkl')

    layer_1, layer_2 = model.hidden_layers[0:2]

    W1 = layer_1.get_weights()
    W2 = layer_2.get_weights()

    top_filters.get_mat_product_viewer(W1, W2)

    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)
    imgs = dataset.get_weights_view(W1.T)

    top_filters.get_connections_viewer(imgs, W1, W2)
