"""
Tests scripts in the DBM folder
"""

import cPickle
import os
import pylearn2.scripts.dbm.show_negative_chains as negative_chains
import pylearn2.scripts.dbm.show_reconstructions as reconstructions
import pylearn2.scripts.dbm.show_samples as samples
from pylearn2.config import yaml_parse
from pylearn2.models.dbm.layer import BinaryVector, BinaryVectorMaxPool
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.dbm.dbm import DBM
from nose.tools import with_setup
from pylearn2.datasets import control


def setup():
    """Create pickle file with a simple model."""
    control.push_load_data(True)
    with open('dbm.pkl', 'wb') as f:
        dataset = MNIST(which_set='train', start=0, stop=300)
        vis_layer = BinaryVector(nvis=784, bias_from_marginals=dataset)
        hid_layer = BinaryVectorMaxPool(layer_name='h', pool_size=1,
                                        irange=.05, init_bias=-2.,
                                        detector_layer_dim=200)
        model = DBM(batch_size=20, niter=1, visible_layer=vis_layer,
                    hidden_layers=[hid_layer])
        model.dataset_yaml_src = """
!obj:pylearn2.datasets.binarizer.Binarizer {
    raw: !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        start: 0,
        stop: 300
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
