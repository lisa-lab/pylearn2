from pylearn2.train import Train
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.training_algorithms.bgd import BGD
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.costs.cost import Cost
import theano.tensor as T
import numpy as np

class SoftmaxModel(Model):
    """A dummy model used for testing.
       Important properties:
           has a parameter (P) for SGD to act on
           has a get_output_space method, so it can tell the
           algorithm what kind of space the targets for supervised
           learning live in
           has a get_input_space method, so it can tell the
           algorithm what kind of space the features live in
    """

    def __init__(self, dim):
        self.dim = dim
        rng = np.random.RandomState([2012,9,25])
        self.P = sharedX( rng.uniform(-1.,1.,(dim,)))

    def get_params(self):
        return [ self.P ]

    def get_input_space(self):
        return VectorSpace(self.dim)

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 2
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X*self.P)



def test_bgd_unsup():

    # tests that we can run the bgd algorithm
    # on an supervised cost.
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25,9,2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)


    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    class DummyCost(Cost):

        def __call__(self, model, X):
            return T.square(model(X)-X).mean()

    cost = DummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = BGD(cost, batch_size=5,
                monitoring_batches=2, monitoring_dataset= monitoring_dataset,
                termination_criterion = termination_criterion)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, extensions=None)

    train.main_loop()

