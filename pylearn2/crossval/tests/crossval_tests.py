import pylearn2.scripts.deep_trainer.run_deep_trainer as dp
import pylearn2.crossval.crossvalidation as cv
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter

def test_holdout_cv():
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)

    train_algo = ExhaustiveSGD(
            learning_rate = 0.1,
            cost=MeanSquaredReconstructionError(),
            batch_size=100,
            monitoring_batches=10,
            monitoring_dataset=trainset,
            termination_criterion=EpochCounter(max_epochs=10),
            update_callbacks=None
            )

    holdoutCV = cv.HoldoutCrossValidation(model=model, algorithm=train_algo, 
        cost=MeanSquaredReconstructionError(),
        dataset=trainset, 
        validation_batch_size=1000,
        train_prop=0.5)

    holdoutCV.crossvalidate_model()
    print "Error: " + str(holdoutCV.get_error())

def test_kfold_cv():
    trainset, testset = dp.get_dataset_cifar10()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)

    train_algo = ExhaustiveSGD(
            learning_rate = 0.1,
            cost=MeanSquaredReconstructionError(),
            batch_size=100,
            monitoring_batches=10,
            monitoring_dataset=trainset,
            termination_criterion=EpochCounter(max_epochs=10),
            update_callbacks=None
            )

    kfoldCV = cv.KFoldCrossValidation(model=model, 
        algorithm=train_algo,
        cost=MeanSquaredReconstructionError(),
        dataset=trainset, 
        validation_batch_size=1000)

    kfoldCV.crossvalidate_model()
    print "Error: " + str(kfoldCV.get_error())

if __name__ == "__main__":
    test_kfold_cv()
