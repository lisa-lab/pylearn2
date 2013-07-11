import itertools

import numpy as np
import theano
T = theano.tensor
F = theano.function

from pylearn2.costs.autoencoder import (MeanBinaryCrossEntropy,
                                        MeanSquaredReconstructionError)
from pylearn2.costs.gsn import GSNCost
from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.datasets.mnist import MNIST
from pylearn2.distributions.parzen import ParzenWindows
from pylearn2.models.gsn import GSN, plushmax
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils import image

HIDDEN_SIZE = 1500
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

WALKBACK = 1

LEARNING_RATE = 0.25
MOMENTUM = 0.5

MAX_EPOCHS = 200
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 100

dataset = MNIST(which_set='train', one_hot=True)

layers = [dataset.X.shape[1], HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(SALT_PEPPER_NOISE)
pre_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)
post_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

mbce = MeanBinaryCrossEntropy()
reconstruction_cost = lambda a, b: mbce.cost(a, b) / 784.0

def test_train_ae():
    gsn = GSN.new_ae(layers, vis_corruptor, pre_corruptor, post_corruptor)
    c = GSNCost([(0, 1.0, reconstruction_cost)], walkback=WALKBACK)
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=c,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
              monitoring_dataset={"test": MNIST(which_set='test')})

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_ae_example.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def test_train_supervised():
    raw_class_cost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: raw_class_cost.cost(a, b) / 784.0

    gsn = GSN.new_classifier(layers + [10], vis_corruptor=vis_corruptor,
                             hidden_pre_corruptor=None, hidden_post_corruptor=None,
                             classifier_act=plushmax)

    # Bugs: works with 1 or 3 hidden layers, but not 2 (theano bugs)
    # cross entropy doesn't work for softmax layer (NaN)
    c = GSNCost(
        [
            (0, 1.0, reconstruction_cost),
            (2, 1.0, classification_cost)
        ],
        walkback=WALKBACK)
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=c,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE
              ,monitoring_dataset={"test": MNIST(which_set='test', one_hot=True)}
              )

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_sup_example.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def test_sample_ae():
    import cPickle
    with open("gsn_ae_example.pkl") as f:
        gsn = cPickle.load(f)

    mb_data = MNIST(which_set='test').X[105:106, :]

    history = gsn.get_samples([(0, mb_data)], walkback=1000,
                              symbolic=False, include_first=True)

    history = list(itertools.chain(*history))
    history = np.vstack(history)

    tiled = image.tile_raster_images(history,
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("gsn_ae_example.png", tiled)

    # code to get log likelihood from kernel density estimator
    # this crashed on GPU (out of memory), but works on CPU
    pw = ParzenWindows(MNIST(which_set='test').X, .20)
    print pw.get_ll(history)

def test_sample_supervised():
    import cPickle
    with open("gsn_sup_example.pkl") as f:
        gsn = cPickle.load(f)

    mb_data = MNIST(which_set='test').X[105:106, :]

    history = gsn.get_samples([(0, mb_data)], walkback=5,
                              symbolic=False, include_first=False,
                              indices=[2])
    history = list(itertools.chain(*history))
    print np.vstack(history)


# some utility methods for viewing MNIST characters without any GUI
def print_char(A):
    print a_to_s(A.round().reshape((28, 28)))

def a_to_s(A):
    """Prints binary array"""
    strs = []
    for row in A:
        x = [None] * len(row)
        for i, num in enumerate(row):
            if num != 0:
                x[i] = "@"
            else:
                x[i] = " "
        strs.append("".join(x))
    return "\n".join(strs)

if __name__ == '__main__':
    test_train_supervised()

