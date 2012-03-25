from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace
from pylearn2.models.model import Model
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from theano import tensor as T


class DummyModel(Model):
    def  __init__(self, num_features):
        self.input_space = VectorSpace(num_features)


class DummyDataset(DenseDesignMatrix):
    def __init__(self, num_examples, num_features):
        rng = np.random.RandomState([4, 12, 17])
        super(DummyDataset, self).__init__(
            X=rng.uniform(1., 2., (num_examples, num_features))
        )


def test_channel_scaling_sequential():
    def channel_scaling_checker(num_examples, mode, num_batches, batch_size):
        num_features = 2
        monitor = Monitor(DummyModel(num_features))
        dataset = DummyDataset(num_examples, num_features)
        monitor.set_dataset(dataset=dataset, mode=mode,
                            num_batches=num_batches, batch_size=batch_size)
        vis_batch = T.matrix()
        mean = vis_batch.mean()
        monitor.add_channel(name='mean', ipt=vis_batch, val=mean)
        monitor()
        assert 'mean' in monitor.channels
        mean = monitor.channels['mean']
        assert len(mean.val_record) == 1
        actual = mean.val_record[0]
        X = dataset.get_design_matrix()
        if batch_size is not None and num_batches is not None:
            total = min(num_examples, num_batches * batch_size)
        else:
            total = num_examples
        expected = X[:total].mean()
        if not np.allclose(expected, actual):
            raise AssertionError("Expected monitor to contain %f but it has "
                                 "%f" % (expected, actual))

    # Specifying num_batches; even split
    yield channel_scaling_checker, 10, 'sequential', 5, None
    # Specifying num_batches; even split
    yield channel_scaling_checker, 10, 'sequential', 2, None
    # Specifying batch_size; even split
    yield channel_scaling_checker, 10, 'sequential', None, 5
    # Specifying batch_size; even split
    yield channel_scaling_checker, 10, 'sequential', None, 2
    # Specifying num_batches; uneven split
    yield channel_scaling_checker, 10, 'sequential', 4, None
    # Specifying num_batches; uneven split
    yield channel_scaling_checker, 10, 'sequential', 3, None
    # Specifying batch_size; uneven split
    yield channel_scaling_checker, 10, 'sequential', None, 3
    # Specifying batch_size; uneven split
    yield channel_scaling_checker, 10, 'sequential', None, 4
    # Specifying both, even split
    yield channel_scaling_checker, 10, 'sequential', 2, 5
    # Specifying both, even split
    yield channel_scaling_checker, 10, 'sequential', 5, 2
    # Specifying both, uneven split, dangling batch
    yield channel_scaling_checker, 10, 'sequential', 3, 4
    # Specifying both, uneven split, non-exhaustive
    yield channel_scaling_checker, 10, 'sequential', 3, 3
