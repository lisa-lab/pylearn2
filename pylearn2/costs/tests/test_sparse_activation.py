"""
Test autoencoder sparse activation cost.
"""
from pylearn2.config import yaml_parse


def test_sparse_activation():
    """Test autoencoder sparse activation cost."""
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()

test_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train
        !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
        {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 10,
            dim: 5,
            num_classes: 2,
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 5,
        nhid: 10,
        act_enc: sigmoid,
        act_dec: linear,
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 5,
        line_search_mode: exhaustive,
        conjugate: 1,
        cost: !obj:pylearn2.costs.cost.SumOfCosts {
            costs: [
              !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
              },
              !obj:pylearn2.costs.autoencoder.SparseActivation {
                  coeff: 0.5,
                  p: 0.2,
              },
            ],
        },
        monitoring_dataset: {
            'train': *train,
        },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter
        {
            max_epochs: 1,
        },
    },
}
"""
