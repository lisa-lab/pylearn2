"""
Test that a smaller version of pylearn2/scripts/tutorials/dbm_demo/rbm.yaml works.

If this file has to be edited, cpylearn2/scripts/tutorials/dbm_demo/rbm.yaml has to be updated
in the same way.

The differences (needed for speed) are:
    * detector_layer_dim: 5 instead of 500
    * monitoring_batches: 2 instead of 10
    * train.stop: 500 
    * max_epochs: 4 instead of 300 
    * MomentumAdjuster: start,stop = 2,3 (instead of 5,6)

This should make the test run in about 2 minutes.
"""
from nose.plugins.skip import SkipTest

from pylearn2.datasets.exc import NoDataPathError
from pylearn2.testing import no_debug_mode

import inspect, os
# Get complete file path for this script
savepath = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),inspect.getfile(inspect.currentframe()))
# Remove '.py'
savepath = savepath[:-3]
#print savepath

@no_debug_mode
def train_dbm():
    train = """
!obj:pylearn2.train.Train {
    dataset: &data !obj:pylearn2.datasets.binarizer.Binarizer {
        raw: &raw_train !obj:pylearn2.datasets.mnist.MNIST {
            which_set: "train",
            one_hot: 1,
            start: 0,
            stop: 500
        }
    },
    model: !obj:pylearn2.models.dbm.DBM {
        batch_size: 100,
        niter: 1,
        visible_layer: !obj:pylearn2.models.dbm.BinaryVector {
            nvis: 784,
            bias_from_marginals: *raw_train,
        },
        hidden_layers: [
            !obj:pylearn2.models.dbm.BinaryVectorMaxPool {
                layer_name: 'h',
                detector_layer_dim: 5,
                pool_size: 1,
                irange: .05,
                init_bias: -2.,
            }
       ]
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
               learning_rate: 1e-3,
               learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                   init_momentum: 0.5,
               },
               monitoring_batches: 2,
               monitoring_dataset : *data,
               cost : !obj:pylearn2.costs.cost.SumOfCosts {
                costs: [
                        !obj:pylearn2.costs.dbm.VariationalPCD {
                           num_chains: 100,
                           num_gibbs_steps: 5
                        },
                        !obj:pylearn2.costs.dbm.WeightDecay {
                          coeffs: [ .0001  ]
                        },
                        !obj:pylearn2.costs.dbm.TorontoSparsity {
                         targets: [ .2 ],
                         coeffs: [ .001 ],
                        }
                       ],
           },
           termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter { max_epochs: 4 },
           update_callbacks: [
                !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
                        decay_factor: 1.000015,
                        min_lr:       0.0001
                }
           ]
        },
    extensions: [
            !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
                final_momentum: .9,
                start: 2,
                saturate: 3
            },
    ],
    save_path: """
    train = train + '"'+savepath
    str_end = """.pkl",
    save_freq : 1
}

"""
    train = train + str_end

    from pylearn2.config import yaml_parse
    train = yaml_parse.load(train)
    train.main_loop()


def test_dbm():
    try:
        train_dbm()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")


if __name__ == '__main__':
    test_dbm()
