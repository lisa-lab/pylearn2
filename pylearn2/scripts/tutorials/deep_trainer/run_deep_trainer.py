#!/usr/bin/env python
from __future__ import print_function

__author__ = "Li Yao"
"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""

MAX_EPOCHS_UNSUPERVISED = 1
MAX_EPOCHS_SUPERVISED = 2

from pylearn2.config import yaml_parse
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.blocks import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train import Train
from optparse import OptionParser

import numpy


class ToyDataset(DenseDesignMatrix):
    def __init__(self):
        # simulated random dataset
        rng = numpy.random.RandomState(seed=42)
        data = rng.normal(size=(1000, 10))
        self.y = numpy.random.binomial(1, 0.5, (1000, 1))
        super(ToyDataset, self).__init__(X=data, y=self.y, y_labels=2)

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

    print('loading CIFAR-10 dataset...')

    # We create the dataset by parsing YAML strings describing the dataset.
    # The yaml parser will automatically tag trainset and testset with a
    # yaml_src field containing the YAML string that was used to specify them.
    # This is useful because later the training algorithm can store this YAML
    # string in the saved model to efficiently describe exactly what data it
    # was trained on.
    template = \
"""!obj:pylearn2.datasets.cifar10.CIFAR10 {
which_set: %s,
center: 1,
rescale: 1,
}"""
    trainset = yaml_parse.load(template % "train")
    testset = yaml_parse.load(template % "test")

    print('...done loading CIFAR-10.')

    return trainset, testset

def get_dataset_mnist():

    print('loading MNIST dataset...')

    # We create the dataset by parsing YAML strings describing the dataset.
    # The yaml parser will automatically tag trainset and testset with a
    # yaml_src field containing the YAML string that was used to specify them.
    # This is useful because later the training algorithm can store this YAML
    # string in the saved model to efficiently describe exactly what data it
    # was trained on.
    template = \
"""!obj:pylearn2.datasets.mnist.MNIST {
which_set: %s,
}"""
    trainset = yaml_parse.load(template % "train")
    testset = yaml_parse.load(template % "test")

    print('...done loading MNIST.')

    return trainset, testset

def get_autoencoder(structure):
    n_input, n_output = structure
    config = {
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'sigmoid',
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
        'act_enc': 'sigmoid',
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

    layer = SoftmaxRegression(n_classes=n_output, irange=0.02, nvis=n_input)

    return layer

def get_layer_trainer_logistic(layer, trainset):
    # configs on sgd

    config = {'learning_rate': 0.1,
              'cost' : Default(),
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': trainset,
              'termination_criterion': EpochCounter(max_epochs=MAX_EPOCHS_SUPERVISED),
              'update_callbacks': None
              }

    train_algo = SGD(**config)
    model = layer
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = None)

def get_layer_trainer_sgd_autoencoder(layer, trainset):
    # configs on sgd
    train_algo = SGD(
            learning_rate = 0.1,
              cost =  MeanSquaredReconstructionError(),
              batch_size =  10,
              monitoring_batches = 10,
              monitoring_dataset =  trainset,
              termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
              update_callbacks =  None
              )

    model = layer
    extensions = None
    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset)

def get_layer_trainer_sgd_rbm(layer, trainset):
    train_algo = SGD(
        learning_rate = 1e-1,
        batch_size =  5,
        #"batches_per_iter" : 2000,
        monitoring_batches =  20,
        monitoring_dataset =  trainset,
        cost = SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        termination_criterion =  EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
        )
    model = layer
    extensions = [MonitorBasedLRAdjuster()]
    return Train(model = model, algorithm = train_algo,
                 save_path='grbm.pkl',save_freq=1,
                 extensions = extensions, dataset = trainset)

def main(args=None):
    """
    args is the list of arguments that will be passed to the option parser.
    The default (None) means use sys.argv[1:].
    """
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="dataset", default="toy",
                      help="specify the dataset, either cifar10, mnist or toy")
    (options,args) = parser.parse_args(args=args)

    if options.dataset == 'toy':
        trainset, testset = get_dataset_toy()
        n_output = 2
    elif options.dataset == 'cifar10':
        trainset, testset, = get_dataset_cifar10()
        n_output = 10

    elif options.dataset == 'mnist':
        trainset, testset, = get_dataset_mnist()
        n_output = 10

    else:
        NotImplementedError()

    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]

    # build layers
    layers = []
    structure = [[n_input, 10], [10, 50], [50, 100], [100, n_output]]
    # layer 0: gaussianRBM
    layers.append(get_grbm(structure[0]))
    # layer 1: denoising AE
    layers.append(get_denoising_autoencoder(structure[1]))
    # layer 2: AE
    layers.append(get_autoencoder(structure[2]))
    # layer 3: logistic regression used in supervised training
    layers.append(get_logistic_regressor(structure[3]))


    # construct training sets for different layers
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

    # unsupervised pretraining
    for i, layer_trainer in enumerate(layer_trainers[0:3]):
        print('-----------------------------------')
        print(' Unsupervised training layer %d, %s'%(i, layers[i].__class__))
        print('-----------------------------------')
        layer_trainer.main_loop()


    print('\n')
    print('------------------------------------------------------')
    print(' Unsupervised training done! Start supervised training...')
    print('------------------------------------------------------')
    print('\n')

    # supervised training
    layer_trainers[-1].main_loop()


if __name__ == '__main__':
    main()
