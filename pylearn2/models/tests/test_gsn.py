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

    trainer = Train(dataset, gsn, algorithm=alg, save_path="gsn_model.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"

def more_tests():
    import cPickle
    with open("gsn_model.pkl") as f:
        gsn = cPickle.load(f)

    print "DONE UNPICKLING"

    # just the first point
    mb = MNIST(which_set='test').X[:1, :]
    x = T.fmatrix()

    print "START GET SAMPLES"
    samples = gsn.get_samples(x, walkback=10)
    print "DONE GET SAMPLES"

    print "START COMPILING"
    f = F([x], samples)
    print "DONE COMPILING"
    f(mb)
    print "DONE COMPUTING"

if __name__ == '__main__':
    more_tests()
