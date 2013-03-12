import cStringIO
import numpy as np

import theano.tensor as T
from theano.tests import disturb_mem


from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import CrossEntropy
from pylearn2.devtools.record import Record
from pylearn2.devtools.record import RecordMode
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.monitor import Monitor
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.testing.cost import CallbackCost
from pylearn2.testing.cost import SumOfParams
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.training_algorithms.sgd import ExponentialDecay
from pylearn2.training_algorithms.sgd import MomentumAdjustor
from pylearn2.training_algorithms.sgd import PolyakAveraging
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils.iteration import _iteration_schemes
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_union
from pylearn2.utils import sharedX

class DummyCost(Cost):
    def __call__(self, model, X, Y = None):
        return T.square(model(X)-X).mean()

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

    # Including a monitoring dataset lets us test that
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
                 save_freq=0, extensions=None)

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


    # Including a monitoring dataset lets us test that
    # the monitor works with unsupervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5


    cost = DummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost, batch_size=5,
                 monitoring_batches=3, monitoring_dataset= monitoring_dataset,
                 termination_criterion=termination_criterion, update_callbacks=None,
                 init_momentum = None, set_batch_size = False)

    train = Train(dataset, model, algorithm, save_path=None,
                 save_freq=0, extensions=None)

    train.main_loop()

def get_topological_dataset(rng, rows, cols, channels, m):
    X = rng.randn(m, rows, cols, channels)

    dim = rows * cols * channels

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m,dim))
    for i in xrange(m):
        Y[i,idx[i]] = 1

    return DenseDesignMatrix(topo_view=X, y=Y)

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

    dataset = get_topological_dataset(rng, rows, cols, channels, m)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    m = 15
    monitoring_dataset = get_topological_dataset(rng, rows, cols, channels, m)

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
                 save_freq=0, extensions=None)

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
                 save_freq=0, extensions=None)

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

    dim = 1
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


def test_determinism():

    # Verifies that running SGD twice results in the same examples getting visited
    # in the same order

    for mode in _iteration_schemes:
        dim = 1
        batch_size = 3
        num_batches = 5
        m = num_batches * batch_size

        dataset = ArangeDataset(m)

        model = SoftmaxModel(dim)

        learning_rate = 1e-3
        batch_size = 5

        visited = [ [ -1 ] * m ]

        def visit(X):
            mx = max(visited[0])
            counter = mx + 1
            for i in X[:,0]:
                i = int(i)
                assert visited[0][i] == -1
                visited[0][i] = counter
                counter += 1


        cost = CallbackCost(visit)

        # We need to include this so the test actually stops running at some point
        termination_criterion = EpochCounter(5)

        def run_algorithm():
            unsupported_modes = ['random_slice', 'random_uniform']
            algorithm =  SGD(learning_rate, cost, batch_size=5,
                   train_iteration_mode = mode,
                     monitoring_dataset=None,
                     termination_criterion=termination_criterion, update_callbacks=None,
                     init_momentum = None, set_batch_size = False)

            algorithm.setup(dataset = dataset, model = model)

            raised = False
            try:
                algorithm.train(dataset)
            except ValueError:
                print mode
                assert mode in unsupported_modes
                raised = True
            if mode in unsupported_modes:
                assert raised
                return True
            return False

        if run_algorithm():
            continue

        visited.insert(0, [ -1 ] * m)

        del model.monitor

        run_algorithm()

        for v in visited:
            assert len(v) == m
            for elem in range(m):
                assert elem in v

        assert len(visited) == 2

        print visited[0]
        print visited[1]
        assert np.all(np.asarray(visited[0]) == np.asarray(visited[1]))

def test_determinism_2():

    """
    A more aggressive determinism test. Tests that apply nodes are all passed inputs
    with the same md5sums, apply nodes are run in same order, etc.
    Uses disturb_mem to try to cause dictionaries to iterate in different orders, etc.
    """

    def run_sgd(mode):
        # Must be seeded the same both times run_sgd is called
        disturb_mem.disturb_mem()
        rng = np.random.RandomState([2012, 11, 27])

        batch_size = 5
        train_batches = 3
        valid_batches = 4
        num_features = 2

        # Synthesize dataset with a linear decision boundary
        w = rng.randn(num_features)

        def make_dataset(num_batches):
            disturb_mem.disturb_mem()
            m = num_batches*batch_size
            X = rng.randn(m, num_features)
            y = np.zeros((m,1))
            y[:,0] = np.dot(X, w) > 0.

            rval =  DenseDesignMatrix(X=X, y=y)

            rval.yaml_src = "" # suppress no yaml_src warning

            X = rval.get_batch_design(batch_size)
            assert X.shape == (batch_size, num_features)

            return rval

        train = make_dataset(train_batches)
        valid = make_dataset(valid_batches)

        num_chunks = 10
        chunk_width = 2
        class ManyParamsModel(Model):
            """
            Make a model with lots of parameters, so that there are many
            opportunities for their updates to get accidentally re-ordered
            non-deterministically. This makes non-determinism bugs manifest
            more frequently.
            """

            def __init__(self):
                self.W1 = [sharedX(rng.randn(num_features, chunk_width)) for i
                    in xrange(num_chunks)]
                disturb_mem.disturb_mem()
                self.W2 = [sharedX(rng.randn(chunk_width)) for i in xrange(num_chunks)]
                self._params = safe_union(self.W1, self.W2)
                self.input_space = VectorSpace(num_features)
                self.output_space = VectorSpace(1)

        disturb_mem.disturb_mem()
        model = ManyParamsModel()
        disturb_mem.disturb_mem()


        class LotsOfSummingCost(Cost):
            """
            Make a cost whose gradient on the parameters involves summing many terms together,
            so that T.grad is more likely to sum things in a random order.
            """

            supervised = True

            def __call__(self, model, X, Y=None, **kwargs):
                disturb_mem.disturb_mem()
                def mlp_pred(non_linearity):
                    Z = [T.dot(X, W) for W in model.W1]
                    H = map(non_linearity, Z)
                    Z = [T.dot(h, W) for h, W in safe_izip(H, model.W2)]
                    pred = sum(Z)
                    return pred

                nonlinearity_predictions = map(mlp_pred, [T.nnet.sigmoid, T.nnet.softplus, T.sqr, T.sin])
                pred = sum(nonlinearity_predictions)
                disturb_mem.disturb_mem()

                return abs(pred-Y[:,0]).sum()

        cost = LotsOfSummingCost()

        disturb_mem.disturb_mem()

        algorithm = SGD(cost=cost,
                batch_size=batch_size,
                init_momentum=.5,
                learning_rate=1e-3,
                monitoring_dataset={'train': train, 'valid':valid},
                update_callbacks=[ExponentialDecay(decay_factor=2., min_lr=.0001)],
                termination_criterion=EpochCounter(max_epochs=5))

        disturb_mem.disturb_mem()

        train_object = Train(
                dataset=train,
                model=model,
                algorithm=algorithm,
                extensions=[
                    PolyakAveraging(start=0),
                    MomentumAdjustor(final_momentum=.9, start=1, saturate=5),
                    ],
                save_freq=0)

        disturb_mem.disturb_mem()

        train_object.main_loop()



    output = cStringIO.StringIO()
    record = Record(file_object=output, replay=False)
    record_mode = RecordMode(record)

    run_sgd(record_mode)

    output = cStringIO.StringIO(output.getvalue())
    playback = Record(file_object=output, replay=True)
    playback_mode = RecordMode(playback)

    run_sgd(playback_mode)

def test_lr_scalers():
    """
    Tests that SGD respects Model.get_lr_scalers
    """

    cost = SumOfParams()

    scales = [ .01, .02, .05, 1., 5. ]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    learning_rate = .001

    class ModelWithScalers(Model):
        def __init__(self):
            self._params = [sharedX(np.zeros(shape)) for shape in shapes]
            self.input_space = VectorSpace(1)

        def get_lr_scalers(self):
            return dict(zip(self._params, scales))

    model = ModelWithScalers()

    dataset = ArangeDataset(1)

    sgd = SGD(cost=cost, learning_rate=learning_rate, init_momentum=0.,
            batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    manual = [param - learning_rate * scale for param, scale in
            zip(manual, scales)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

    manual = [param - learning_rate * scale for param, scale in
            zip(manual, scales)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

def test_lr_scalers_momentum():
    """
    Tests that SGD respects Model.get_lr_scalers when using
    momentum.
    """

    cost = SumOfParams()

    scales = [ .01, .02, .05, 1., 5. ]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    learning_rate = .001

    class ModelWithScalers(Model):
        def __init__(self):
            self._params = [sharedX(np.zeros(shape)) for shape in shapes]
            self.input_space = VectorSpace(1)

        def get_lr_scalers(self):
            return dict(zip(self._params, scales))

    model = ModelWithScalers()

    dataset = ArangeDataset(1)

    momentum = 0.5

    sgd = SGD(cost=cost, learning_rate=learning_rate, init_momentum=momentum,
            batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    inc = [ - learning_rate * scale for param, scale in
            zip(manual, scales)]
    manual = [param + i for param, i in zip(manual, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

    manual = [param - learning_rate * scale + i * momentum for param, scale, i in
            zip(manual, scales, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

if __name__ == '__main__':
    test_lr_scalers()
    test_lr_scalers_momentum()
