"""
.. todo::

    WRITEME
"""
from __future__ import print_function

import cPickle as pickle
import itertools

import numpy as np
from theano.compat.six.moves import xrange
import theano.tensor as T

from pylearn2.expr.activations import rescaled_softmax
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.gsn import GSNCost
from pylearn2.corruption import (BinomialSampler, GaussianCorruptor,
                                 MultinomialSampler, SaltPepperCorruptor,
                                 SmoothOneHotCorruptor)
from pylearn2.datasets.mnist import MNIST
from pylearn2.distributions.parzen import ParzenWindows
from pylearn2.models.gsn import GSN, JointGSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD, MonitorBasedLRAdjuster
from pylearn2.utils import image, safe_zip

# define some common parameters
HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 0.5

WALKBACK = 0

LEARNING_RATE = 0.25
MOMENTUM = 0.75

MAX_EPOCHS = 100
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 100

ds = MNIST(which_set='train')

def test_train_ae():
    """
    .. todo::

        WRITEME
    """
    GC = GaussianCorruptor

    gsn = GSN.new(
        layer_sizes=[ds.X.shape[1], 1000],
        activation_funcs=["sigmoid", "tanh"],
        pre_corruptors=[None, GC(1.0)],
        post_corruptors=[SaltPepperCorruptor(0.5), GC(1.0)],
        layer_samplers=[BinomialSampler(), None],
        tied=False
    )

    # average MBCE over example rather than sum it
    _mbce = MeanBinaryCrossEntropy()
    reconstruction_cost = lambda a, b: _mbce.cost(a, b) / ds.X.shape[1]

    c = GSNCost([(0, 1.0, reconstruction_cost)], walkback=WALKBACK)

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=10
   )

    trainer = Train(ds, gsn, algorithm=alg, save_path="gsn_ae_example.pkl",
                    save_freq=5)
    trainer.main_loop()
    print("done training")

def test_sample_ae():
    """
    Visualize some samples from the trained unsupervised GSN.
    """
    with open("gsn_ae_example.pkl") as f:
        gsn = pickle.load(f)

    # random point to start at
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
    print(pw.get_ll(history))

def test_train_supervised():
    """
    Train a supervised GSN.
    """
    # initialize the GSN
    gsn = GSN.new(
        layer_sizes=[ds.X.shape[1], 1000, ds.y.shape[1]],
        activation_funcs=["sigmoid", "tanh", rescaled_softmax],
        pre_corruptors=[GaussianCorruptor(0.5)] * 3,
        post_corruptors=[SaltPepperCorruptor(.3), None, SmoothOneHotCorruptor(.5)],
        layer_samplers=[BinomialSampler(), None, MultinomialSampler()],
        tied=False
    )

    # average over costs rather than summing
    _rcost = MeanBinaryCrossEntropy()
    reconstruction_cost = lambda a, b: _rcost.cost(a, b) / ds.X.shape[1]

    _ccost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: _ccost.cost(a, b) / ds.y.shape[1]

    # combine costs into GSNCost object
    c = GSNCost(
        [
            # reconstruction on layer 0 with weight 1.0
            (0, 1.0, reconstruction_cost),

            # classification on layer 2 with weight 2.0
            (2, 2.0, classification_cost)
        ],
        walkback=WALKBACK,
        mode="supervised"
    )

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=10,
    )

    trainer = Train(ds, gsn, algorithm=alg,
                    save_path="gsn_sup_example.pkl", save_freq=10,
                    extensions=[MonitorBasedLRAdjuster()])
    trainer.main_loop()
    print("done training")

def test_classify():
    """
    See how well a (supervised) GSN performs at classification.
    """
    with open("gsn_sup_example.pkl") as f:
        gsn = pickle.load(f)

    gsn = JointGSN.convert(gsn)

    # turn off corruption
    gsn._corrupt_switch = False

    ds = MNIST(which_set='test')
    mb_data = ds.X
    y = ds.y

    for i in xrange(1, 10):
        y_hat = gsn.classify(mb_data, trials=i)
        errors = np.abs(y_hat - y).sum() / 2.0

        # error indices
        #np.sum(np.abs(y_hat - y), axis=1) != 0

        print(i, errors, errors / mb_data.shape[0])

def test_sample_supervised(idxs=None, noisy=True):
    """
    Visualize samples and labels produced by GSN.
    """
    with open("gsn_sup_example.pkl") as f:
        gsn = pickle.load(f)

    gsn._corrupt_switch = noisy

    ds = MNIST(which_set='test')

    if idxs is None:
        data = ds.X[100:150]
    else:
        data = ds.X[idxs]

    # change the walkback parameter to make the data fill up rows in image
    samples = gsn.get_samples([(0, data)],
                              indices=[0, 2],
                              walkback=21, symbolic=False,
                              include_first=True)
    stacked = vis_samples(samples)
    tiled = image.tile_raster_images(stacked,
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("gsn_sup_example.png", tiled)

def vis_samples(samples):
    """
    .. todo::

        WRITEME
    """
    from PIL import ImageDraw, ImageFont
    img = image.pil_from_ndarray(np.zeros((28, 28)))

    chains = []
    num_rows = samples[0][0].shape[0]

    for row in xrange(num_rows):
        images = []
        labels = []

        for step in samples:
            assert len(step) == 2

            images.append(step[0][row])

            vec = step[1][row]
            sorted_idxs = np.argsort(vec)
            label1 = sorted_idxs[-1]
            label2 = sorted_idxs[-2]

            if vec[label1] == 0:
                ratio = 1
            else:
                ratio = vec[label2] / vec[label1]

            c = img.copy()
            draw = ImageDraw.Draw(c)
            draw.text((8, 11), str(label1), 255)
            draw.text((14, 11), str(label2), int(255 * ratio))
            nd = image.ndarray_from_pil(c)[:, :, 0]
            nd = nd.reshape((1, 784))
            labels.append(nd)

        data = safe_zip(images, labels)
        data = list(itertools.chain(*data))

        # white block to indicate end of chain
        data.extend([np.ones((1, 784))] * 2)

        chains.append(np.vstack(data))

    return np.vstack(chains)


# some utility methods for viewing MNIST characters without any GUI
def print_char(A):
    """
    .. todo::

        WRITEME
    """
    print(a_to_s(A.round().reshape((28, 28))))

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
    test_classify()
    test_sample_supervised()
