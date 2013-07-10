import copy

import numpy as np
import theano
T = theano.tensor
F = theano.function

from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.gsn import GSNCost
from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils import image

HIDDEN_SIZE = 1500
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

WALKBACK = 4

LEARNING_RATE = 0.25 / 784
MOMENTUM = 0.5 / 784

MAX_EPOCHS = 200
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 100

dataset = MNIST(which_set='train')

layers = [dataset.X.shape[1], HIDDEN_SIZE, HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(SALT_PEPPER_NOISE)
pre_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)
post_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

gsn = GSN.new_ae(layers, vis_corruptor, pre_corruptor, post_corruptor)

def test_train():
    c = GSNCost([(0, 1.0, MeanBinaryCrossEntropy())], walkback=WALKBACK)
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=c,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE
              ,monitoring_dataset={"test": MNIST(which_set='test')}
              )

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_repro.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def test_sample():
    import cPickle
    with open("gsn_model6.pkl") as f:
        gsn = cPickle.load(f)

    mb = T.fmatrix()
    f_init = F([mb], gsn._set_activations(mb))

    prev = T.fmatrices(len(gsn.activations))
    f_step = F(prev, gsn._update(copy.copy(prev)),
               on_unused_input='ignore')

    precor = T.fmatrices(len(gsn.activations))
    evens = xrange(0, len(gsn.activations), 2)
    f_even_corrupt = F(
        precor,
        gsn._apply_postact_corruption(
            copy.copy(precor),
            evens
        ),
    )

    mb_data = MNIST(which_set='test').X[105:106, :]

    history = [mb_data[0]]
    activations = f_init(mb_data)

    for _ in xrange(2500):
        activations = f_step(*activations)
        history.append(activations[0][0])
        activations = f_even_corrupt(*activations)

    tiled = image.tile_raster_images(np.array(history),
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("woot_test.png", tiled)

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
    test_train()
