import cPickle as pickle
import itertools

import numpy as np
import theano
T = theano.tensor
F = theano.function

from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.gsn import *
from pylearn2.corruption import *
from pylearn2.datasets.mnist import MNIST
from pylearn2.distributions.parzen import ParzenWindows
from pylearn2.models.gsn import *
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD, MonitorBasedLRAdjuster
from pylearn2.utils import image, safe_zip, identity

HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 0.5

WALKBACK = 0

LEARNING_RATE = 0.3
MOMENTUM = 0.5

MAX_EPOCHS = 100
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
              termination_criterion=EpochCounter(5),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
              monitoring_dataset={"test": MNIST(which_set='test')})

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_ae_example.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def test_sample_ae():
    with open("gsn_ae_example.pkl") as f:
        gsn = pickle.load(f)

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

def test_train_supervised():
    raw_class_cost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: raw_class_cost.cost(a, b) / 10.0

    dc = DropoutCorruptor(.5)
    gc = GaussianCorruptor(1.0)
    dgc = ComposedCorruptor(dc, gc)

    gsn = GSN.new([784, 1000, 600, 300, 10],
                  ["sigmoid", "tanh", "tanh", "tanh", plushmax],
                  [gc, None, None, None, gc],
                  [SaltPepperCorruptor(0.3), dc, dc, dgc, SmoothOneHotCorruptor(0.75)],
                  [BinomialSampler(), None, None, None, MultinomialSampler()],
                  tied=False)

    c1 = GSNCost(
        [
            (0, 1.0, reconstruction_cost),
            (4, 3.0, classification_cost)
        ],
        walkback=1, mode='supervised')

    c2 = GSNCost(
        [
            (0, 3.0, reconstruction_cost),
            (4, 1.0, classification_cost)
        ],
        walkback=1, mode='anti_supervised')

    algs = map(lambda c:
        SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=c,
            termination_criterion=EpochCounter(1),
            batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
            monitoring_dataset=MNIST(which_set='train', one_hot=True),
            monitoring_batches=10, monitor_iteration_mode="shuffled_sequential"
        ),
        [c1, c2]
    )

    monitors = [None, None]
    for i in xrange(50):
        print "ITERATION %s" % i
        for step in xrange(2):
            alg = algs[step]
            if i != 0:
                gsn.monitor = monitors[step]

            trainer = Train(dataset, gsn, algorithm=alg,
                            save_path="gsn_sup_example.pkl",
                            extensions=[MonitorBasedLRAdjuster()])
            trainer.main_loop()

            if step == 0:
                alg.termination_criterion = EpochCounter(5)
            else:
                alg.termination_criterion = EpochCounter(1)


            monitors[step] = gsn.monitor
            del gsn.monitor


        if i % 5 == 0:
            trainer.save()

    print "done training"

def test_train_supervised2():
    raw_class_cost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: raw_class_cost.cost(a, b) / 10.0

    dc = DropoutCorruptor(.5)
    gc = GaussianCorruptor(1.0)
    dgc = ComposedCorruptor(dc, gc)
    x = [None] * 3

    gsn = GSN.new([784, 1000, 10],
                  ["sigmoid", "tanh", plushmax],
                  [gc, None, gc],
                  [SaltPepperCorruptor(0.3), dgc, SmoothOneHotCorruptor(0.75)],
                  [BinomialSampler(), None, MultinomialSampler()],
                  tied=False)

    c = CrazyGSNCost(
        [
            (0, 1.0, crazy_costf),
            (2, 3.0, crazy_costf)
        ],
        walkback=1, p_keep=[.5, .3])

    alg = SGD(
        LEARNING_RATE, init_momentum=MOMENTUM, cost=c,
        termination_criterion=EpochCounter(100),
        batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
        monitoring_dataset=MNIST(which_set='train', one_hot=True),
        monitoring_batches=10, monitor_iteration_mode="shuffled_sequential"
    )

    trainer = Train(dataset, gsn, algorithm=alg,
                    save_path="gsn_sup_example.pkl", save_freq=5,
                    extensions=[MonitorBasedLRAdjuster()])
    trainer.main_loop()
    print "done training"


def test_classify():
    with open("gsn_sup_example.pkl") as f:
        gsn = pickle.load(f)

    gsn = JointGSN.convert(gsn, 0, 2)
    gsn._corrupt_switch = False

    #ds = MNIST(which_set='test', one_hot=True)
    ds = MNIST(which_set='train', one_hot=True)
    mb_data = ds.X
    y = ds.y

    for i in xrange(1, 10):
        y_hat = gsn.classify(mb_data, trials=i)
        errors = np.abs(y_hat - y).sum() / 2.0

        # error indices
        #return np.sum(np.abs(y_hat - y), axis=1) != 0

        print i, errors, errors / 10000.0

def test_sample_supervised(idxs=None, noisy=True):
    with open("gsn_sup_example.pkl") as f:
        gsn = pickle.load(f)

    if not noisy:
        gsn._corrupt_switch = False

    #ds = MNIST(which_set='test', one_hot=True)
    ds = MNIST(which_set='train', one_hot=True)

    if idxs is None:
        data = ds.X[:50]
    else:
        data = ds.X[idxs]

    samples = gsn.get_samples([(0, data)],
                              indices=[0, 4],
                              walkback=19, symbolic=False,
                              include_first=True)
    stacked = vis_samples(samples)
    tiled = image.tile_raster_images(stacked,
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("gsn_sup_example.png", tiled)

def vis_samples(samples):
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
    test_train_supervised2()
