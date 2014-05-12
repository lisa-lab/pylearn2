"""
Tests scripts in the DBM folder
"""

import cPickle
import pylearn2.scripts.dbm.show_negative_chains as negative_chains
from pylearn2.models.dbm.layer import BinaryVector, BinaryVectorMaxPool
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.dbm.dbm import DBM


def test_show_negative_chains():
    """Test the show_negative_chains script main function"""

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
    raw: &raw_train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: "train",
        start: 0
    }
}
"""
        model.layer_to_chains = model.make_layer_to_state(1)
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    negative_chains.show_negative_chains('dbm.pkl')
