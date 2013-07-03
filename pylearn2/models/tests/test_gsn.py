import numpy as np
import theano
T = theano.tensor
F = theano.function

from pylearn2.costs.gsn import MBWalkbackCrossEntropy as Cost
from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils import image

HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

LEARNING_RATE = 0.25 / 784
MOMENTUM = 0.5 / 784

MAX_EPOCHS = 80
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 32

WALKBACK = 0

dataset = MNIST(which_set='train')

layers = [dataset.X.shape[1], HIDDEN_SIZE, HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(0.4)
pre_corruptor = GaussianCorruptor(.2)
post_corruptor = GaussianCorruptor(.2)

gsn = GSN.new(layers, vis_corruptor, pre_corruptor, post_corruptor)

def debug(walkback = 0):

    check = lambda x: np.any(np.isnan(x)) or np.any(np.isinf(x))
    check_val = lambda x: np.all(x > 0.0) and np.all(x < 1.0)

    x = T.fmatrix()
    mb = dataset.X[:4, :]

    gsn._set_activations(x)
    data = F([x], gsn.activations)(mb)
    print "Activation shapes: ", data[0].shape, data[1].shape

    data = F([x], gsn.activations)(mb)
    print "STEP 0"
    for j in xrange(len(data)):
        print "Activation %d: " % j, data[j][0][:5]
    print map(check, data)

    for time in xrange(1, len(gsn.aes) + walkback + 1):
        print ''
        gsn._update(time=time)
        data = F([x], gsn.activations)(mb)
        print "DATA GOOD: ", check_val(data[0])
        print "STEP (PRE CORRUPTION) %d" % time
        for j in xrange(len(data)):
            print "Activation %d: " % j, data[j][0][:5]


        print "STEP (POST CORRUPTION) %d" % time
        gsn._apply_postact_corruption(xrange(0, len(gsn.activations), 2), 2*time)
        data = F([x], gsn.activations)(mb)
        if time > len(gsn.aes):
            print "WALKBACK %d" % (time - len(gsn.aes))

        for j in xrange(len(data)):
            print "Activation %d: " % j, data[j][0][:5]

        print map(check, data)

def test():
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=Cost(walkback=WALKBACK),
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE
              ,monitoring_dataset={"test": MNIST(which_set='test')}
              )

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_model2.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def sampling_test():
    import cPickle
    with open("gsn_model.pkl") as f:
        gsn = cPickle.load(f)

    mb = T.fmatrix()
    f_init = F([mb], gsn._set_activations(mb))

    prev = T.fmatrices(len(gsn.activations))
    f_step = F(prev, gsn._update(prev))

    evens = xrange(0, len(gsn.activations), 2)
    corrupted = gsn._apply_postact_corruption(prev, evens)
    f_even_corrupt = F(prev, corrupted)


    mb_data = MNIST(which_set='test').X[:1, :]
    activations = f_init(mb_data)
    history = [activations[0][0]]
    for _ in xrange(1000):
        activations = f_step(*activations)
        history.append(activations[0][0])
        activations = f_even_corrupt(*activations)

    tiled = image.tile_raster_images(np.array(history),
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("woot_test.png", tiled)

if __name__ == '__main__':
    sampling_test()
