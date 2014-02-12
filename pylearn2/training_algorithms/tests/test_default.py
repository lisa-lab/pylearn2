import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.models.s3c import S3C, E_Step, Grad_M_Step


def test_multiple_monitoring_datasets():
    # tests that DefaultTrainingAlgorithm can take multiple monitoring datasets.

    BATCH_SIZE = 2
    BATCHES = 3
    NUM_FEATURES = 4
    dim = 3
    m = 10

    rng = np.random.RandomState([2014,02,25])
    X = rng.randn(m, dim)
    Y = rng.randn(m, dim)

    train = DenseDesignMatrix(X=X)
    test = DenseDesignMatrix(X=Y)

    algorithm = DefaultTrainingAlgorithm(batch_size = BATCH_SIZE,
            batches_per_iter = BATCHES, monitoring_dataset = {'train': train, 'test':test})

    model = S3C( nvis = NUM_FEATURES, nhid = 1,
            irange = .01, init_bias_hid = 0., init_B = 1.,
            min_B = 1., max_B = 1., init_alpha = 1.,
            min_alpha = 1., max_alpha = 1., init_mu = 0.,
            m_step = Grad_M_Step( learning_rate = 0.),
            e_step = E_Step( h_new_coeff_schedule = [ 1. ]))

    algorithm.setup(model = model, dataset = train)
    algorithm.train(dataset = train)

if __name__ == '__main__':
    test_multiple_monitoring_datasets()
    
   


