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
import cStringIO
from pylearn2.devtools.record import Record
from pylearn2.devtools.record import RecordMode
from pylearn2.devtools import disturb_mem
from pylearn2.utils import safe_union
from pylearn2.utils import safe_izip

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

def test_determinism():

    """
    Tests that apply nodes are all passed inputs
    with the same md5sums, apply nodes are run in same order, etc.
    Uses disturb_mem to try to cause dictionaries to iterate in different orders, etc.
    """

    def run_bgd(mode):
        # Must be seeded the same both times run_bgd is called
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

        algorithm = BGD(cost=cost,
                batch_size=batch_size,
                updates_per_batch=5,
                scale_step=.5,
                conjugate=1,
                reset_conjugate=0,
                monitoring_dataset={'train': train, 'valid':valid},
                termination_criterion=EpochCounter(max_epochs=5))

        disturb_mem.disturb_mem()

        train_object = Train(
                dataset=train,
                model=model,
                algorithm=algorithm,
                save_freq=0)

        disturb_mem.disturb_mem()

        train_object.main_loop()



    output = cStringIO.StringIO()
    record = Record(file_object=output, replay=False)
    record_mode = RecordMode(record)

    run_bgd(record_mode)

    output = cStringIO.StringIO(output.getvalue())
    playback = Record(file_object=output, replay=True)
    playback_mode = RecordMode(playback)

    run_bgd(playback_mode)

