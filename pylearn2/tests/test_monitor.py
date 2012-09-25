from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace
from pylearn2.models.model import Model
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
import numpy as np
from theano import tensor as T
from pylearn2.models.s3c import S3C, E_Step, Grad_M_Step
from nose.plugins.skip import SkipTest


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
        monitor.add_dataset(dataset=dataset, mode=mode,
                            num_batches=num_batches, batch_size=batch_size)
        vis_batch = T.matrix()
        mean = vis_batch.mean()
        monitor.add_channel(name='mean', ipt=vis_batch, val=mean, dataset=dataset)
        try:
            monitor()
        except NotImplementedError:
            # make sure this was due to the unimplemented batch_size case
            if num_batches is None:
                assert num_examples % batch_size != 0
            else:
                assert num_examples % num_batches != 0
            raise SkipTest()
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

def test_counting():
    BATCH_SIZE = 2
    BATCHES = 3
    NUM_FEATURES = 4
    num_examples = BATCHES * BATCH_SIZE
    dataset = DummyDataset( num_examples = num_examples,
            num_features = NUM_FEATURES)
    algorithm = DefaultTrainingAlgorithm( batch_size = BATCH_SIZE,
            batches_per_iter = BATCHES)
    model = S3C( nvis = NUM_FEATURES, nhid = 1,
            irange = .01, init_bias_hid = 0., init_B = 1.,
            min_B = 1., max_B = 1., init_alpha = 1.,
            min_alpha = 1., max_alpha = 1., init_mu = 0.,
            m_step = Grad_M_Step( learning_rate = 0.),
            e_step = E_Step( h_new_coeff_schedule = [ 1. ]))
    algorithm.setup(model = model, dataset = dataset)
    algorithm.train(dataset = dataset)
    if not ( model.monitor.get_batches_seen() == BATCHES):
        raise AssertionError('Should have seen '+str(BATCHES) + \
                ' batches but saw '+str(model.monitor.get_batches_seen()))

    assert model.monitor.get_examples_seen() == num_examples
    assert isinstance(model.monitor.get_examples_seen(),int)
    assert isinstance(model.monitor.get_batches_seen(),int)
