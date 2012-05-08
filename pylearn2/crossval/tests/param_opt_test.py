import pylearn2.crossval.param_optimization as opt
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
import pylearn2.scripts.deep_trainer.run_deep_trainer as dp


# Testing the kfold and hold-out cross-validation using the grid-search
# and random search algorithms as a hyperparameter optimization method.
def main():
    # We will train a denoising autoencoder.
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)
    # Specify the hyperparameters that will not be fixed throughout the
    # cross-validation. Each hyperparameter is an object of type `Hyperparam`.
    # We specify for each of the hyperparameters, the list of
    # values or the range of values the hyperparameter can take on. The data
    # type and the number of grid points (if the grid search algorithm will be
    # eventually used) are also specified for each hyperparameter.
    import numpy as np
    var_hyperparams = [
                  opt.Hyperparam('learning_rate', opt.TypeHyperparam.LIST, [0.1], 1, float),
                  opt.Hyperparam('batch_size', opt.TypeHyperparam.LIST, [100], 1, int),
                  opt.Hyperparam('termination_criterion', opt.TypeHyperparam.LIST,
                                 [EpochCounter(max_epochs=10),
                                 EpochCounter(max_epochs=20)])]
    # The list of variable hyperparameters is then given to a
    # VariableHyperparams constructor. The VariableHyperparams class is used
    # for modifiying the variable hyperparmeters just defined.
    variable_params = opt.VariableHyperparams(var_hyperparams)
    # For instance, we can get the Hyperparam object associated with the
    # learning_rate hyperparameter.
    hyperparameter = variable_params.get_hyperparam('learning_rate')
    # We can also set a range of values for the learning rate.
    variable_params.reset_hyperparam('learning_rate',
                                     type_param=opt.TypeHyperparam.RANGE,
                                     values=[0.001, 0.1],
                                     n_grid_points=4)
    # We define next the hyperparameters fixed values as a dict where the name
    # of the hyperparameter (key) is associated with its fixed value.
    fixed_hyperparams = {'cost': MeanSquaredReconstructionError(),
                         'update_callbacks': None,
                         'monitoring_dataset': trainset
                        }
    # We instantiate the class HyperparamOptimization.
    param_optimizer = opt.HyperparamOptimization(model=model, dataset=trainset,
                            algorithm=ExhaustiveSGD,
                            var_hyperparams=variable_params,
                            fixed_hyperparams=fixed_hyperparams,
                            cost=MeanSquaredReconstructionError(),
                            validation_batch_size=1000, train_prop=0.5,
                            )
    # We run the kfold crossvalidation on the DAE model using the random search
    # algorithm. We specify 5 combinations of hyperparameters to test.
    param_optimizer(crossval_mode='kfold',
                    search_mode='random_search',
                    n_combinations=5)
    # Once the crossvalidation finishes, we can collect the N best experiments
    # based on the cost.
    best_exps = param_optimizer.get_topN_exp(3)

    # The hold-out crossvalidation is now tested on the DAE model using the
    # grid search algorithm. The number of grid points along each hyperparameter
    # axis was defined earlier when defining each hyperparameter.
    # Since the HyperparamOptimization constructor's parameters didn't change, we
    # can use the same HyperparamOptimization instance.
    param_optimizer(crossval_mode='holdout',
                    search_mode='grid_search')
    # Once the crossvalidation finishes, we can collect the N best experiments
    # based on the cost.
    best_exps = param_optimizer.get_topN_exp(3)

if __name__ == '__main__':
    main()