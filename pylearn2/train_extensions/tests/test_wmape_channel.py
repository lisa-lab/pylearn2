"""
Tests for WMAPE.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn
from theano.compile import function
import numpy as np
from numpy.testing import assert_allclose


def test_wmape():
    """Test WMapeChannel."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()

    X = trainer.model.get_input_space().make_theano_batch()
    Y = trainer.model.fprop(X)

    f = function([X], Y, allow_input_downcast=True)
    y_hat = f(trainer.dataset.X)

    wmape_num_exp = abs(trainer.dataset.y - y_hat).sum()
    wmape_den_exp = abs(trainer.dataset.y).sum()
    exp_array = np.asarray([wmape_num_exp, wmape_den_exp])

    wmape_num_real = trainer.model.monitor.channels['train_wmape_num'].\
        val_record
    wmape_den_real = trainer.model.monitor.channels['train_wmape_den'].\
        val_record
    real_array = np.asarray([wmape_num_real[-1], wmape_den_real[-1]])

    assert_allclose(exp_array, real_array)


test_yaml = """
!obj:pylearn2.train.Train {
    dataset:
      &train !obj:pylearn2.testing.datasets.\
random_dense_design_matrix_for_regression
      {
          rng: !obj:numpy.random.RandomState { seed: 1 },
          num_examples: 10,
          dim: 10,
          reg_min: 1,
          reg_max: 1000
      },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: h0,
                dim: 10,
                irange: 0.05,
            },
            !obj:pylearn2.models.mlp.Linear {
                layer_name: y,
                dim: 1,
                irange: 0.,
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        monitoring_dataset: {
            'train': *train,
        },
        batches_per_iter: 1,
        monitoring_batches: 1,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: train_wmape_num,
                    prop_decrease: 0.,
                    N: 1,
                },
            ],
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.wmape_channel.WMapeNumeratorChannel {},
        !obj:pylearn2.train_extensions.wmape_channel.\
WMapeDenominatorChannel {},
    ],
}
"""
