#!/usr/bin/python
"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""

MAX_EPOCHS_UNSUPERVISED = 1
MAX_EPOCHS_SUPERVISED = 2

from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.models.mlp import Softmax, MLP
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.datasets import cifar10
from pylearn2.datasets import mnist
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.costs.cost import MethodCost
from pylearn2.train import Train
from pylearn2.space import VectorSpace
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
    
    train_path = 'cifar10_train.pkl'
    test_path = 'cifar10_test.pkl'

    if os.path.exists(train_path) and \
            os.path.exists(test_path):
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        testset = serial.load(test_path)
        
    else:
        print 'loading raw data...'
        trainset = cifar10.CIFAR10(which_set="train", one_hot=True)
        testset =  cifar10.CIFAR10(which_set="test", one_hot=True)

        serial.save('cifar10_train.pkl', trainset)
        serial.save('cifar10_test.pkl', testset)

        # this path will be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        testset.yaml_src = '!pkl: "%s"' % test_path
    
    return trainset, testset

def get_dataset_mnist():
    
    train_path = 'mnist_train.pkl'
    test_path = 'mnist_test.pkl'

    if os.path.exists(train_path) and \
            os.path.exists(test_path):
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        testset = serial.load(test_path)
        
    else:
        print 'loading raw data...'
        trainset = mnist.MNIST(which_set="train", one_hot=True)
        testset =  mnist.MNIST(which_set="test", one_hot=True)

        serial.save('mnist_train.pkl', trainset)
        serial.save('mnist_test.pkl', testset)

        # this path will be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        testset.yaml_src = '!pkl: "%s"' % test_path
    
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

    last_layer = Softmax(n_classes=n_output, irange=0.02, layer_name='softmax')

    layer = MLP([last_layer], nvis=n_input)
    
    return layer

def get_layer_trainer_logistic(layer, trainset):
    # configs on sgd
    
    config = {'learning_rate': 0.1,
              'cost' : MethodCost('cost_from_X', supervised=True),
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

def main():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="dataset", default="toy",
                      help="specify the dataset, either cifar10, mnist or toy")
    (options,args) = parser.parse_args()

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
    for i, layer_trainer in enumerate(layer_trainers[0:3]):
        print '-----------------------------------'
        print ' Unsupervised training layer %d, %s'%(i, layers[i].__class__)
        print '-----------------------------------'
        layer_trainer.main_loop()

        
    print '\n'
    print '------------------------------------------------------'
    print ' Unsupervised training done! Start supervised training...'
    print '------------------------------------------------------'
    print '\n'
    
    #supervised training
    layer_trainers[-1].main_loop() 


if __name__ == '__main__':
    main()
