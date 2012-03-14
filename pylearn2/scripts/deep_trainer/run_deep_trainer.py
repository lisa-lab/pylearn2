#!/usr/bin/python
"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""

MAX_EPOCHS = 1
SAVE_MODEL = False

from pylearn2.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.datasets import cifar10
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedTermCrit
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.training_callbacks.training_callback import TrainingCallback
from pylearn2.costs.supervised_cost import CrossEntropy
from pylearn2.scripts.train import Train
import pylearn2.utils.serial as serial
import sys
import os
import datetime
from optparse import OptionParser


import numpy
import numpy.random


class ToyDataset(DenseDesignMatrix):
    def __init__(self):

        # simulated random dataset
        rng = numpy.random.RandomState(seed=42)
        data = rng.normal(size=(1000, 10))
        self.y = numpy.ones((1000, 2))
        positive = numpy.random.binomial(1, 0.5, [1000])
        self.y[:,0]=positive
        self.y[:,1]=1-positive
        super(ToyDataset, self).__init__(X=data, y=self.y)


class ModelSaver(TrainingCallback):
    """
    Save model after the last epoch.
    The saved models then may be visualized.
    """

    def __init__(self):
        self.current_epoch = 0

    def __call__(self, model, dataset, algorithm):
        if SAVE_MODEL is True:
            save_path = 'cifar10_grbm' + str(self.current_epoch) + '_epoch.pkl'
            save_start = datetime.datetime.now()
            serial.save(save_path, model)
            save_end = datetime.datetime.now()
            delta = (save_end - save_start)
            print 'saving model...done. saving took', str(delta)

        self.current_epoch += 1

def get_dataset_toy():
    """
    The toy dataset is only meant to used for testing pipelines.
    Do not try to visualize weights on it. It is not picture and
    has no color channel info to support visualization
    """
    trainset = ToyDataset()
    testset = ToyDataset()
    return trainset, testset

def get_dataset_cifar10():
    """
    The orginal pipeline on cifar10 from pylearn2. Please refer to
    pylearn2/scripts/train_example/make_dataset.py for details.
    """

    train_path = 'cifar10_preprocessed_train.pkl'
    test_path = 'cifar10_preprocessed_test.pkl'

    if os.path.exists(train_path) and \
            os.path.exists(test_path):
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        testset = serial.load(test_path)
    else:
        print 'loading raw data...'
        trainset = cifar10.CIFAR10(which_set="train")
        testset =  cifar10.CIFAR10(which_set="test")

        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()

        pipeline.items.append(
            preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000))

        pipeline.items.append(preprocessing.GlobalContrastNormalization())

        pipeline.items.append(preprocessing.ZCA())

        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        trainset.use_design_loc('train_design.npy')

        testset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        testset.use_design_loc('test_design.npy')

        print 'saving preprocessed data...'
        serial.save('cifar10_preprocessed_train.pkl', trainset)
        serial.save('cifar10_preprocessed_test.pkl', testset)

        trainset.yaml_src = '!pkl: "%s"' % train_path
        testset.yaml_src = '!pkl: "%s"' % test_path

    # this path will be used for visualizing weights after training is done
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

def get_layer_trainer_logistic(layer, trainset):
    # configs on sgd
    config = {'learning_rate': 0.1,
              'cost' : CrossEntropy(),
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': None,
              'termination_criterion': EpochCounter(max_epochs=10),
              'update_callbacks': None
              }

    train_algo = ExhaustiveSGD(**config)
    model = layer
    callbacks = None
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            callbacks = callbacks)

def get_layer_trainer_sgd_autoencoder(layer, trainset):
    # configs on sgd
    train_algo = ExhaustiveSGD(
            learning_rate = 0.1,
              cost =  MeanSquaredReconstructionError(),
              batch_size =  10,
              monitoring_batches = 10,
              monitoring_dataset =  None,
              termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS),
              update_callbacks =  None
              )

    model = layer
    callbacks = None
    return Train(model = model,
            algorithm = train_algo,
            callbacks = callbacks,
            dataset = trainset)

def get_layer_trainer_sgd_rbm(layer, trainset):
    train_algo = ExhaustiveSGD(
        learning_rate = 1e-1,
        batch_size =  5,
        #"batches_per_iter" : 2000,
        monitoring_batches =  20,
        monitoring_dataset =  trainset,
        cost = SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        termination_criterion =  EpochCounter(max_epochs=MAX_EPOCHS),
        # another option:
        # MonitorBasedTermCrit(prop_decrease=0.01, N=10),
        )
    model = layer
    callbacks = [MonitorBasedLRAdjuster(), ModelSaver()]
    return Train(model = model, algorithm = train_algo,
            callbacks = callbacks, dataset = trainset)

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


    #construct training sets for different layers
    trainset = [ trainset ,
                TransformerDataset( raw = trainset, transformer = layers[0] ),
                TransformerDataset( raw = trainset, transformer = StackedBlocks( layers[0:2] )),
                TransformerDataset( raw = trainset, transformer = StackedBlocks( layers[0:3] ))  ]

    # construct layer trainers
    layer_trainers = []
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[0], trainset[0]))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[1], trainset[1]))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[2], trainset[2]))
    layer_trainers.append(get_layer_trainer_logistic(layers[3], trainset[3]))

    #unsupervised pretraining
    for layer_trainer in layer_trainers[0:3]:
        layer_trainer.main_loop()

    #supervised training
    layer_trainers[-1].main_loop()


if __name__ == '__main__':
    main()
