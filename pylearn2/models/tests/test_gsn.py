import theano

#from pylearn2.costs.gsn import MSWalkbackReconstructionError as Cost
#from pylearn2.costs.gsn import MBWalkbackCrossEntropy as Cost
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError as Cost

from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

HIDDEN_SIZE = 1500
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 2

LEARNING_RATE = 0.25
MOMENTUM = 0.5

MAX_EPOCHS = 20
BATCHES_PER_EPOCH = 10

BATCH_SIZE = 32

dataset = MNIST(which_set='train')

# just 1 hidden layer
layers = [dataset.X.shape[1], HIDDEN_SIZE]

vis_corruptor = SaltPepperCorruptor(0.5)
pre_corruptor = None
post_corruptor = None

gsn = GSN.new(layers, vis_corruptor, pre_corruptor, post_corruptor)

def debug():
    import theano.tensor as T
    F = theano.function

    x = T.fmatrix()
    mb = dataset.X[:4, :]

    gsn._set_activations(x)
    data = F([x], gsn.activations)(mb)
    print "Activation shapes: ", data[0].shape, data[1].shape

    # update odd
    gsn._update_activations(xrange(1, len(gsn.activations), 2))
    data = F([x], gsn.activations)(mb)
    print "ODD UPDATES"
    print "Activation 0: ", data[0]
    print "Activation 1: ", data[1]

    # update even
    gsn._update_activations(xrange(0, len(gsn.activations), 2))
    data = F([x], gsn.activations)(mb)
    print "EVEN UPDATES"
    print "Activation 0: ", data[0]
    print "Activation 1: ", data[1]

def test():
    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=Cost(),
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE)

    trainer = Train(dataset, gsn, algorithm=alg)
    trainer.main_loop()
    print "done training"

if __name__ == '__main__':
    test()
