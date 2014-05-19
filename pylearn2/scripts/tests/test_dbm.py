"""
Tests scripts in the DBM folder
"""

import cPickle
import os
import pylearn2.scripts.dbm.show_negative_chains as negative_chains
import pylearn2.scripts.dbm.show_reconstructions as reconstructions
import pylearn2.scripts.dbm.show_samples as samples
import pylearn2.scripts.dbm.top_filters as top_filters
from pylearn2.config import yaml_parse
from pylearn2.models.dbm.layer import BinaryVector, BinaryVectorMaxPool
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.dbm.dbm import DBM
from nose.tools import with_setup
from pylearn2.datasets import control
from pylearn2.utils import serial


def setup():
    """Create pickle file with a simple model."""
    control.push_load_data(True)
    with open('dbm.pkl', 'wb') as f:
        dataset = MNIST(which_set='train', start=0, stop=100)
        vis_layer = BinaryVector(nvis=784, bias_from_marginals=dataset)
        hid_layer1 = BinaryVectorMaxPool(layer_name='h1', pool_size=1,
                                         irange=.05, init_bias=-2.,
                                         detector_layer_dim=50)
        hid_layer2 = BinaryVectorMaxPool(layer_name='h2', pool_size=1,
                                         irange=.05, init_bias=-2.,
                                         detector_layer_dim=10)
        model = DBM(batch_size=20, niter=1, visible_layer=vis_layer,
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

    model = reconstructions.load_model('dbm.pkl', m)
    dataset = reconstructions.load_dataset(model.dataset_yaml_src,
                                           use_test_set='n')

    recons_viewer = reconstructions.ReconsViewer(model, dataset, rows, cols)
    recons_viewer.update_viewer()


@with_setup(setup, teardown)
def test_show_samples():
    """Test the samples update_viewer function"""
    rows = 10
    cols = 10
    m = rows * cols

    model = samples.load_model('dbm.pkl', m)
    dataset = yaml_parse.load(model.dataset_yaml_src)

    samples_viewer = samples.SamplesViewer(model, dataset, rows, cols)
    samples_viewer.update_viewer()


@with_setup(setup, teardown)
def test_top_filters():
    model = serial.load('dbm.pkl')

    layer_1, layer_2 = model.hidden_layers[0:2]

    W1 = layer_1.get_weights()
    W2 = layer_2.get_weights()

    top_filters.get_mat_product_viewer(W1, W2)

    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)
    imgs = dataset.get_weights_view(W1.T)

    top_filters.get_connections_viewer(imgs, W1, W2)
