from pylearn2.train import Train
from pylearn2.monitor import Monitor
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.costs.cost import CrossEntropy
from pylearn2.costs.cost import Cost
import theano.tensor as T
import numpy as np
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.testing.cost import CallbackCost

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

class TopoSoftmaxModel(Model):
    """A dummy model used for testing.
       Like SoftmaxModel but its features have 2 topological
       dimensions. This tests that the training algorithm
       will provide topological data correctly.
    """

    def __init__(self, rows, cols, channels):
        dim = rows * cols * channels
        self.input_space = Conv2DSpace((rows, cols), channels)
        self.dim = dim
        rng = np.random.RandomState([2012,9,25])
        self.P = sharedX( rng.uniform(-1.,1.,(dim,)))

    def get_params(self):
        return [ self.P ]

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 4
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X.reshape((X.shape[0],self.dim))*self.P)

def test_sgd_unspec_num_mon_batch():

    # tests that if you don't specify a number of
    # monitoring batches, SGD configures the monitor
    # to run on all the data

    m = 25

    visited = [ False ] * m
    rng = np.random.RandomState([25,9,2012])
    X = np.zeros((m,1))
    X[:,0] = np.arange(m)
    dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(1)

    learning_rate = 1e-3
    batch_size = 5

    class DummyCost(Cost):

        def __call__(self, model, X):
            return T.square(model(X)-X).mean()

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=None, monitoring_dataset=dataset,
                 termination_criterion=None, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    algorithm.setup(dataset = dataset, model = model)

    monitor = Monitor.get_monitor(model)

    X = T.matrix()

    def tracker(X, y):
        assert y is None
        assert X.shape[1] == 1
        for i in xrange(X.shape[0]):
            visited[int(X[i,0])] = True

    monitor.add_channel(name = 'tracker',
            ipt = X, val = 0., prereqs = [ tracker ])

    monitor()

    if False in visited:
        print visited
        assert False
def test_sgd_sup():

    # tests that we can run the sgd algorithm
    # on a supervised cost.
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25,9,2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X, y=Y)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = CrossEntropy()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=3, monitoring_dataset= monitoring_dataset,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, callbacks=None)

    train.main_loop()

def test_sgd_unsup():

    # tests that we can run the sgd algorithm
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

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=3, monitoring_dataset= monitoring_dataset,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, callbacks=None)

    train.main_loop()

def test_sgd_topo():

    # tests that we can run the sgd algorithm
    # on data with topology
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    rows = 3
    cols = 4
    channels = 2
    dim = rows * cols * channels
    m = 10

    rng = np.random.RandomState([25,9,2012])

    X = rng.randn(m, rows, cols, channels)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    dataset = DenseDesignMatrix(topo_view=X, y=Y)

    m = 15
    X = rng.randn(m, rows, cols, channels)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(topo_view=X, y=Y)

    model = TopoSoftmaxModel(rows, cols, channels)

    learning_rate = 1e-3
    batch_size = 5

    cost = CrossEntropy()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=3, monitoring_dataset= monitoring_dataset,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, callbacks=None)

    train.main_loop()


def test_sgd_no_mon():

    # tests that we can run the sgd algorithm
    # wihout a monitoring dataset
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25,9,2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = CrossEntropy()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_dataset=None,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, callbacks=None)

    train.main_loop()


def test_reject_mon_batch_without_mon():

    # tests that setting up the sgd algorithm
    # without a monitoring dataset
    # but with monitoring_batches specified is an error

    dim = 3
    m = 10

    rng = np.random.RandomState([25,9,2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = CrossEntropy()

    try:
        algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=3, monitoring_dataset=None,
                 update_callbacks=None,
                 init_momentum = None, set_batch_size = False)
    except ValueError:
        return

    assert False



def test_sgd_sequential():

    # tests that requesting train_iteration_mode = 'sequential'
    # works

    dim = 2
    batch_size = 3
    m = 5 * batch_size

    dataset = ArangeDataset(m)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    visited = [ False ] * m

    def visit(X):
        assert X.shape[1] == 1
        assert np.all(X[1:] == X[0:-1]+1)
        start = int(X[0,0])
        if start > 0:
            assert visited[start - 1]
        for i in xrange(batch_size):
            assert not visited[start+i]
            visited[start+i] = 1


    cost = CallbackCost(visit)

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost, batch_size=5,
                train_iteration_mode = 'sequential',
                 monitoring_dataset=None,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    algorithm.setup(dataset = dataset, model = model)

    algorithm.train(dataset)

    assert all(visited)


