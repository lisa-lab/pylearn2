"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""

from pylearn2.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.training_algorithms.sgd import UnsupervisedExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.datasets import cifar10
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedTermCrit
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.training_callbacks.training_callback import TrainingCallback
from pylearn2.cost import SquaredError
import pylearn2.utils.serial as serial
import sys
import datetime
from optparse import OptionParser

from deep_trainer import LayerTrainer, DeepTrainer

import numpy
import numpy.random

SAVE_MODEL = False

class ToyDataset(DenseDesignMatrix):
    def __init__(self):

        # simulated random dataset
        rng = numpy.random.RandomState(seed=42)
        data = rng.normal(size=(1000, 10))
        self.y = numpy.random.binomial(1, 0.5, [1000, 1])
        super(ToyDataset, self).__init__(data)


class ModelSaver(TrainingCallback):
    """
    Save model after the last epoch.
    The saved models then may be visualized.
    """
    def __call__(self, model, dataset, algorithm, current_epoch):
        #import pdb;pdb.set_trace()
        if SAVE_MODEL is True:
            #save_path = 'cifar_grbm_smd_' + str(current_epoch) + '_epoch.pkl'
            save_path = 'cifar10_grbm' + str(current_epoch) + '_epoch.pkl'
            save_start = datetime.datetime.now()
            global YAML
            model.dataset_yaml_src = YAML
            serial.save(save_path, model)
            save_end = datetime.datetime.now()
            delta = (save_end - save_start)
            print 'saving model...done. saving took', str(delta)

def get_dataset_toy():
    """
    The toy dataset is only meant to used for testing pipelines.
    Do not try to visualize weights on it. It is not picture and
    has no color channel info to support visualization
    """
    global YAML
    YAML = ""
    trainset = ToyDataset()
    testset = ToyDataset()
    return trainset, testset

def get_dataset_cifar10():
    """
    The orginal pipeline on cifar10 from pylearn2. Please refer to
    pylearn2/scripts/train_example/make_dataset.py for details.
    """
    print 'loading data...'

    trainset = cifar10.CIFAR10(which_set="train")
    testset =  cifar10.CIFAR10(which_set="test")
    #import pdb;pdb.set_trace()
    #serial.save('cifar10_unpreprocessed_train.pkl', trainset)

    pipeline = preprocessing.Pipeline()

    pipeline.items.append(
        preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000))

    pipeline.items.append(preprocessing.GlobalContrastNormalization())

    pipeline.items.append(preprocessing.ZCA())

    trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    trainset.use_design_loc('train_design.npy')

    testset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    testset.use_design_loc('train_design.npy')

    serial.save('cifar10_preprocessed_train.pkl', trainset)

    # this path will be used for visualizing weights after training is done
    global YAML
    YAML = '!pkl: "cifar10_preprocessed_train.pkl"'
    return trainset, testset

def get_autoencoder(structure):
    n_input, n_output = structure
    config = {
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': 'sigmoid',
        'irange': 0.001,
    }
    return Autoencoder(**config)

def get_denoising_autoencoder(structure):
    n_input, n_output = structure
    curruptor = BinomialCorruptor(corruption_level=0.5)
    config = {
        'corruptor': curruptor,
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': 'sigmoid',
        'irange': 0.001,
    }
    return DenoisingAutoencoder(**config)

def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "energy_function_class" : GRBM_Type_1,
        "learn_sigma" : True,
        "init_sigma" : .4,
        "init_bias_hid" : -2.,
        "mean_vis" : False,
        "sigma_lr_scale" : 1e-3
        }

    return GaussianBinaryRBM(**config)

def get_logistic_regressor(structure):
    n_input, n_output = structure

    return LogisticRegressionLayer(nvis=n_input, nclasses=n_output)

def get_layer_trainer_logistic(layer, testset):
    # configs on sgd
    config = {'learning_rate': 0.1,
              'cost' : SquaredError(),
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': None,
              'termination_criterion': EpochCounter(max_epochs=10),
              'update_callbacks': None
              }

    train_algo = UnsupervisedExhaustiveSGD(**config)
    model = layer
    callbacks = None
    return LayerTrainer(model, train_algo, callbacks, testset)

def get_layer_trainer_sgd_autoencoder(layer, testset):
    # configs on sgd
    config = {'learning_rate': 0.1,
              'cost' : MeanSquaredReconstructionError(),
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': None,
              'termination_criterion': EpochCounter(max_epochs=50),
              'update_callbacks': None
              }

    train_algo = UnsupervisedExhaustiveSGD(**config)
    model = layer
    callbacks = None
    return LayerTrainer(model, train_algo, callbacks, testset)

def get_layer_trainer_sgd_rbm(layer, testset):
    config = {
        "learning_rate" : 1e-1,
        "batch_size" : 5,
        #"batches_per_iter" : 2000,
        "monitoring_batches" : 20,
        "monitoring_dataset" : None,
        "cost": SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        "termination_criterion": EpochCounter(max_epochs=50),
        # another option:
        # MonitorBasedTermCrit(prop_decrease=0.01, N=10),
        "update_callbacks": MonitorBasedLRAdjuster()
        }
    train_algo = UnsupervisedExhaustiveSGD(**config)
    model = layer
    callbacks = [ModelSaver()]
    return LayerTrainer(model, train_algo, callbacks, testset)

def main():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="dataset", default="toy",
                      help="specify the dataset, either cifar10 or toy")
    (options,args) = parser.parse_args()

    global SAVE_MODEL

    if options.dataset == 'toy':
        trainset, testset = get_dataset_toy()
        SAVE_MODEL = False
    elif options.dataset == 'cifar10':
        trainset, testset, = get_dataset_cifar10()
        SAVE_MODEL = True

    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]

    # build layers
    layers = []
    structure = [[n_input, 400], [400, 50], [50, 100], [100, 2]]
    # layer 0: gaussianRBM
    layers.append(get_grbm(structure[0]))
    # layer 1: denoising AE
    layers.append(get_denoising_autoencoder(structure[1]))
    # layer 2: AE
    layers.append(get_autoencoder(structure[2]))
    # layer 3: logistic regression used in supervised training
    layers.append(get_logistic_regressor(structure[3]))

    # construct layer trainers
    layer_trainers = []
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[0], testset))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[1], testset))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[2], testset))
    layer_trainers.append(get_layer_trainer_logistic(layers[3], testset))

    # init trainer that performs
    master_trainer = DeepTrainer(trainset, layer_trainers)

    # unsupervised pretraining
    layers_to_unsupervised_train = [0, 1, 2]
    master_trainer.train_unsupervised(layers_to_unsupervised_train)

    #layers_to_supervised_train = [0, 1, 2, 3]
    #master_trainer.train_supervised()

if __name__ == '__main__':
    main()
