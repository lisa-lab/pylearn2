from pylearn2.corruption import GaussianCorruptor, SaltPepperCorruptor
from pylearn2.costs.gsn import MBWalkbackCrossEntropy as Cost
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

def test():
    HIDDEN_SIZE = 1500
    SALT_PEPPER_NOISE = 0.4
    GAUSSIAN_NOISE = 2

    LEARNING_RATE = 0.25
    MOMENTUM = 0.5

    MAX_EPOCHS = 100
    BATCHES_PER_EPOCH = 20

    layers = [0, HIDDEN_SIZE, HIDDEN_SIZE]
    vis_corruptor = SaltPepperCorruptor(SALT_PEPPER_NOISE)
    pre_corruptor = post_corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

    gsn = GSN.new(layers, vis_corruptor, pre_corruptor, post_corruptor)

    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=Cost,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH)

    dataset = MNIST(which_set='train')

    Train(dataset, gsn, algorithm=alg)
