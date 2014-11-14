import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.rbm import RBM
from pylearn2.models.s3c import S3C, E_Step, Grad_M_Step
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.training_algorithms.training_algorithm import NoBatchSizeError


def test_multiple_monitoring_datasets():
    # tests that DefaultTrainingAlgorithm can take multiple
    # monitoring datasets.

    BATCH_SIZE = 1
    BATCHES = 3
    dim = 4
    m = 10

    rng = np.random.RandomState([2014, 2, 25])
    X = rng.randn(m, dim)
    Y = rng.randn(m, dim)

    train = DenseDesignMatrix(X=X)
    test = DenseDesignMatrix(X=Y)

    algorithm = DefaultTrainingAlgorithm(
        batch_size=BATCH_SIZE,
        batches_per_iter=BATCHES,
        monitoring_dataset={'train': train, 'test': test})

    model = S3C(nvis=dim, nhid=1,
                irange=.01, init_bias_hid=0., init_B=1.,
                min_B=1., max_B=1., init_alpha=1.,
                min_alpha=1., max_alpha=1., init_mu=0.,
                m_step=Grad_M_Step(learning_rate=0.),
                e_step=E_Step(h_new_coeff_schedule=[1.]))

    algorithm.setup(model=model, dataset=train)
    algorithm.train(dataset=train)


def test_unspecified_batch_size():

    # Test that failing to specify the batch size results in a
    # NoBatchSizeError

    m = 1
    dim = 2
    rng = np.random.RandomState([2014, 3, 17])
    X = rng.randn(m, dim)
    train = DenseDesignMatrix(X=X)

    rbm = RBM(nvis=dim, nhid=3)
    trainer = DefaultTrainingAlgorithm()
    try:
        trainer.setup(rbm, train)
    except NoBatchSizeError:
        return
    raise AssertionError("Missed the lack of a batch size")


if __name__ == '__main__':
    test_multiple_monitoring_datasets()
