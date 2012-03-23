from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace
from pylearn2.models.model import Model
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from theano import tensor as T

NVIS = 3
BATCHES = 4
BATCH_SIZE = 5
NUM_EXAMPLES = BATCHES * BATCH_SIZE

class DummyModel(Model):
    input_space = VectorSpace(NVIS)

class DummyDataset(DenseDesignMatrix):
    def __init__(self):
        rng = np.random.RandomState([4,12,17])
        DenseDesignMatrix.__init__(self, X = rng.uniform(1.,2.,(NUM_EXAMPLES,NVIS)))

def test_channel_scaling():
    monitor = Monitor( DummyModel())
    dataset = DummyDataset()
    monitor.set_dataset( dataset = dataset, batches = BATCHES, batch_size = BATCH_SIZE)
    V = T.matrix()
    m = V.mean()

    monitor.add_channel(name = 'mean', ipt = V, val = m)

    monitor()

    assert 'mean' in monitor.channels
    mean = monitor.channels['mean']

    assert len(mean.val_record) == 1
    actual = mean.val_record[0]

    X = dataset.get_design_matrix()
    expected = X.mean()

    if not np.allclose(expected,actual):
        raise AssertionError("Expected monitor to contain %d but it has %d",expected,actual)

