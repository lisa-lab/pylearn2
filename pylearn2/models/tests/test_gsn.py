import copy

import numpy as np
import theano
T = theano.tensor
F = theano.function

#from pylearn2.costs.gsn import MSWalkbackReconstructionError as Cost
#from pylearn2.costs.gsn import MBWalkbackCrossEntropy as Cost
#from pylearn2.costs.autoencoder import MeanSquaredReconstructionError as Cost
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy as Cost

from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

LEARNING_RATE = 0.25
MOMENTUM = 0.5

MAX_EPOCHS = 20
BATCHES_PER_EPOCH = 1000
BATCH_SIZE = 32

dataset = MNIST(which_set='train')

# just 1 hidden layer
layers = [dataset.X.shape[1], HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(0.1)
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
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=Cost(),
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE
              ,monitoring_dataset=dataset
              )

    trainer = Train(dataset, gsn, algorithm=alg)
    trainer.main_loop()
    print "done training"


if __name__ == '__main__':
    test()
