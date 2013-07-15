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
from pylearn2.utils import image, safe_zip

HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

WALKBACK = 0

LEARNING_RATE = 0.25
MOMENTUM = 0.5

MAX_EPOCHS = 200
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 100

dataset = MNIST(which_set='train', one_hot=True)

layers = [dataset.X.shape[1], HIDDEN_SIZE, HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(SALT_PEPPER_NOISE)
pre_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)
post_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

mbce = MeanBinaryCrossEntropy()
reconstruction_cost = lambda a, b: mbce.cost(a, b) / 784.0
reconstruction_cost = mbce

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

    gsn = GSN.new_classifier(layers + [10], vis_corruptor=vis_corruptor,
                             hidden_pre_corruptor=None, hidden_post_corruptor=None,
                             classifier_act=plushmax)

    # Bugs: works with 1 or 3 hidden layers, but not 2 (theano bugs)
    # cross entropy doesn't work for softmax layer (NaN)
    c = GSNCost(
        [
            (0, 1.0, reconstruction_cost),
            (3, 1.0, classification_cost)
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

#####################
# tests and utilities
#####################
def debug():
    raw_class_cost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: raw_class_cost.cost(a, b) / 784.0
    classification_cost = MeanBinaryCrossEntropy()

    gsn = GSN.new_classifier(layers + [10], vis_corruptor=vis_corruptor,
                             hidden_pre_corruptor=None, hidden_post_corruptor=None,
                             classifier_act=plushmax)

    _costf = lambda t, o: T.mean(T.nnet.binary_crossentropy(o, t))
    _t, _o = T.fmatrices(2)
    costf = F([_t, _o], _costf(_t, _o))

    cost = GSNCost(
        [
            (0, 1.0, _costf),
            (3, 1.0, _costf)
        ],
        walkback=WALKBACK)

    mb_data = MNIST(which_set='test').X[105:106, :]
    y = MNIST(which_set='test', one_hot=True).y[105:106, :]
    data = (mb_data, y)

    _x = T.fmatrix()
    _y = T.fmatrix()
    get_cost = F([_x, _y], cost.expr(gsn, (_x, _y)))
    z = get_cost(mb_data, y)

    # copy of code within cost
    layer_idxs = [idx for idx, _, _ in cost.costs]
    output = gsn.get_samples(safe_zip(layer_idxs, data),
                             walkback=0, indices=layer_idxs, symbolic=False)
    total = 0.0
    for cost_idx in xrange(len(cost.costs)):
        for step in output:
            total += costf(data[cost_idx], step[cost_idx])
    # end copy

    samples = gsn.get_samples(safe_zip(layer_idxs, (_x, _y)), indices=layer_idxs)
    vis = [s[0] for s in samples]
    soft = [s[1] for s in samples]

    get_vis = F([_x, _y], vis)
    get_soft = F([_x, _y], soft)

    from IPython import embed; embed()

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
    debug()
