__authors__ = "Carlo D'Eramo, Francesco Visin, Matteo Matteucci"
__copyright__ = "Copyright 2014-2015, Politecnico di Milano"
__credits__ = ["Carlo D'Eramo, Francesco Visin, Matteo Matteucci"]
__license__ = "3-clause BSD"
__maintainer__ = "AIR-lab"
__email__ = "carlo.deramo@mail.polimi.it, francesco.visin@polimi.it, \
matteo.matteucci@polimi.it"

import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.datasets import mnist_augmented
from theano import function

# PARAMETERS
N_HIDDEN_0 = 500
N_HIDDEN_1 = 1000

PRETRAINING = 1
TRAINING = 1
FINETUNING = 1

# PRETRAINING
MAX_EPOCHS_L1 = 100
MAX_EPOCHS_L2 = 200

# TRAINING
MAX_EPOCHS_DBM = 500
SOFTMAX = 0

# FINETUNING
MAX_EPOCHS_MLP = 500
DROPOUT = 0
MF_STEPS = 1  # mf_steps for data augmentation


@no_debug_mode
def test_train_example():

    # path definition
    cwd = os.getcwd()
    train_path = cwd  # training path is the current working directory
    try:
        os.chdir(train_path)

        # START PRETRAINING
        # load and train first layer
        train_yaml_path = os.path.join(train_path, 'dbm_mnist_l1.yaml')
        layer1_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l1 = {'train_stop': 60000,
                           'batch_size': 100,
                           'monitoring_batches': 5,
                           'nhid': N_HIDDEN_0,
                           'max_epochs': MAX_EPOCHS_L1,
                           'save_path': train_path
                           }

        if PRETRAINING:

            layer1_yaml = layer1_yaml % (hyper_params_l1)
            train = yaml_parse.load(layer1_yaml)

            print '\n-----------------------------------'
            print '     Unsupervised pre-training'
            print '-----------------------------------\n'

            print '\nPre-Training first layer...\n'
            train.main_loop()

        # load and train second layer
        train_yaml_path = os.path.join(train_path, 'dbm_mnist_l2.yaml')
        layer2_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l2 = {'train_stop': 60000,
                           'batch_size': 100,
                           'monitoring_batches': 5,
                           'nvis': hyper_params_l1['nhid'],
                           'nhid': N_HIDDEN_1,
                           'max_epochs': MAX_EPOCHS_L2,
                           'save_path': train_path
                           }

        if PRETRAINING:

            layer2_yaml = layer2_yaml % (hyper_params_l2)
            train = yaml_parse.load(layer2_yaml)

            print '\n...Pre-training second layer...\n'
            train.main_loop()

        if TRAINING:

            # START TRAINING
            if SOFTMAX:
                train_yaml_path = os.path.join(train_path,
						'dbm_mnist_softmax.yaml')
            else:
                train_yaml_path = os.path.join(train_path, 'dbm_mnist.yaml')
            yaml = open(train_yaml_path, 'r').read()
            hyper_params_dbm = {'train_stop': 60000,
                                'valid_stop': 60000,
                                'batch_size': 100,
                                'detector_layer_1_dim':hyper_params_l1['nhid'],
                                'detector_layer_2_dim':hyper_params_l2['nhid'],
                                'monitoring_batches': 5,
                                'max_epochs': MAX_EPOCHS_DBM,
                                'save_path': train_path
                                }

            yaml = yaml % (hyper_params_dbm)
            train = yaml_parse.load(yaml)

            rbm1 = serial.load(os.path.join(train_path, 'dbm_mnist_l1.pkl'))
            rbm2 = serial.load(os.path.join(train_path, 'dbm_mnist_l2.pkl'))
            pretrained_rbms = [rbm1, rbm2]

            # clamp pretrained weights into respective dbm layers
            for h, l in zip(train.model.hidden_layers, pretrained_rbms):
                h.set_weights(l.get_weights())

            # clamp pretrained biases into respective dbm layers
            bias_param = pretrained_rbms[0].get_params()[1]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.visible_layer.set_biases(bias)
            bias_param = pretrained_rbms[-1].get_params()[1]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.hidden_layers[0].set_biases(bias)
            bias_param = pretrained_rbms[-1].get_params()[2]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.hidden_layers[1].set_biases(bias)

            print '\nAll layers weights and biases have been clamped\
	    to the respective layers of the DBM'

            print '\n-----------------------------------'
            print '     Unsupervised training'
            print '-----------------------------------\n'

            print '\nTraining phase...'
            train.main_loop()

        if FINETUNING:

            # START SUPERVISED TRAINING WITH BACKPROPAGATION
            print '\n-----------------------------------'
            print '       Supervised training'
            print '-----------------------------------'

            # load dbm as a mlp
            if DROPOUT:
                train_yaml_path = os.path.join(train_path, 
				'dbm_mnist_mlp_dropout.yaml')
            else:
                train_yaml_path = os.path.join(train_path, 
				'dbm_mnist_mlp.yaml')
            mlp_yaml = open(train_yaml_path, 'r').read()
            hyper_params_mlp = {'train_stop': 60000,
                                'valid_stop': 60000,
                                'batch_size': 5000,
                                'nvis': 784 + hyper_params_l2['nhid'],
                                'n_h0': hyper_params_l1['nhid'],
                                'n_h1': hyper_params_l2['nhid'],
                                'max_epochs': MAX_EPOCHS_MLP,
                                'save_path': train_path
                                }

            mlp_yaml = mlp_yaml % (hyper_params_mlp)
            train = yaml_parse.load(mlp_yaml)

            if SOFTMAX:
                dbm = serial.load(os.path.join(train_path, 
				'dbm_mnist_softmax.pkl'))
            else:
                dbm = serial.load(os.path.join(train_path, 
				'dbm_mnist.pkl'))

            train.dataset = mnist_augmented.MNIST_AUGMENTED(
					dataset=train.dataset,
                                        which_set='train',
					one_hot=1,
					model=dbm, start=0,
					stop=hyper_params_mlp['train_stop'],
					mf_steps=MF_STEPS)
            train.algorithm.monitoring_dataset = {
		# 'valid' : mnist_augmented.MNIST_AUGMENTED(
			# dataset=train.algorithm.monitoring_dataset['valid'],
			# which_set='train', one_hot=1, model=dbm,
			# start=hyper_params_mlp['train_stop'],
			# stop=hyper_params_mlp['valid_stop'],
			# mf_steps=mf_steps), 
                'test' : mnist_augmented.MNIST_AUGMENTED(
			dataset=train.algorithm.monitoring_dataset['test'],
			which_set='test', one_hot=1, model=dbm,
			mf_steps=MF_STEPS)}

            # DBM TRAINED WEIGHTS CLAMPED FOR FINETUNING AS
	    # EXPLAINED BY HINTON

            # concatenate weights between first and second hidden
	    # layer & weights between visible and first hidden layer
            train.model.layers[0].set_weights(numpy.concatenate((
				dbm.hidden_layers[1].get_weights().transpose(),
                                dbm.hidden_layers[0].get_weights())))

            # then clamp all the others normally
            for l, h in zip(train.model.layers[1:], dbm.hidden_layers[1:]):
                l.set_weights(h.get_weights())

            # clamp biases       
            for l, h in zip(train.model.layers, dbm.hidden_layers):
                l.set_biases(h.get_biases())

            print '\nDBM trained weights and biases have been
	          clamped in the MLP.'

            print '\n...Finetuning...\n'
            train.main_loop()

    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
