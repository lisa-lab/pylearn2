from theano.compat.six.moves import cStringIO, xrange
import numpy as np

import theano.tensor as T
from theano.tests import disturb_mem
from theano.tests.record import Record, RecordMode
from theano import shared

from pylearn2.train import Train
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import sharedX
from pylearn2.training_algorithms.bgd import BGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.cost import Cost
from pylearn2.utils import safe_union
from pylearn2.utils import safe_izip
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils import function
from pylearn2.costs.cost import FixedVarDescr
from pylearn2.costs.cost import SumOfCosts

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
        super(SoftmaxModel, self).__init__()
        self.dim = dim
        rng = np.random.RandomState([2012,9,25])
        self.P = sharedX( rng.uniform(-1.,1.,(dim,)))
        self.force_batch_size = None

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

        def expr(self, model, data):
            self.get_data_specs(model)[0].validate(data)
            X = data
            return T.square(model(X) - X).mean()

        def get_data_specs(self, model):
            return (model.get_input_space(), model.get_input_source())

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
    Uses disturb_mem to try to cause dictionaries to iterate in different
    orders, etc.
    """

    def run_bgd(mode):
        # Must be seeded the same both times run_bgd is called
        disturb_mem.disturb_mem()
        rng = np.random.RandomState([2012, 11, 27, 8])

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
                super(ManyParamsModel, self).__init__()
                self.W1 = [sharedX(rng.randn(num_features, chunk_width)) for i
                    in xrange(num_chunks)]
                disturb_mem.disturb_mem()
                self.W2 = [sharedX(rng.randn(chunk_width)) for i in
                        xrange(num_chunks)]
                self._params = safe_union(self.W1, self.W2)
                self.input_space = VectorSpace(num_features)
                self.output_space = VectorSpace(1)

        disturb_mem.disturb_mem()
        model = ManyParamsModel()
        disturb_mem.disturb_mem()


        class LotsOfSummingCost(Cost):
            """
            Make a cost whose gradient on the parameters involves summing many
            terms together,
            so that T.grad is more likely to sum things in a random order.
            """

            supervised = True

            def expr(self, model, data, **kwargs):
                self.get_data_specs(model)[0].validate(data)
                X, Y = data
                disturb_mem.disturb_mem()
                def mlp_pred(non_linearity):
                    Z = [T.dot(X, W) for W in model.W1]
                    H = [non_linearity(z) for z in Z]
                    Z = [T.dot(h, W) for h, W in safe_izip(H, model.W2)]
                    pred = sum(Z)
                    return pred

                nonlinearity_predictions = map(mlp_pred, [T.nnet.sigmoid,
                    T.nnet.softplus, T.sqr, T.sin])
                pred = sum(nonlinearity_predictions)
                disturb_mem.disturb_mem()

                return abs(pred-Y[:,0]).sum()

            def get_data_specs(self, model):
                data = CompositeSpace((model.get_input_space(),
                                       model.get_output_space()))
                source = (model.get_input_source(), model.get_target_source())
                return (data, source)

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



    output = cStringIO()
    record = Record(file_object=output, replay=False)
    record_mode = RecordMode(record)

    run_bgd(record_mode)

    output = cStringIO(output.getvalue())
    playback = Record(file_object=output, replay=True)
    playback_mode = RecordMode(playback)

    run_bgd(playback_mode)


def test_fixed_vars():

    """
    A very basic test of the the fixed vars interface.
    Checks that the costs' expr and get_gradients methods
    are called with the right parameters and that the updates
    functions are called the right number of times.
    """

    """
    Notes: this test is fairly messy. PL made some change to how
    FixedVarDescr worked. FixedVarDescr got an added data_specs
    field. But BGD itself was never changed to obey this data_specs.
    Somehow these tests passed regardless. It looks like PL just built
    a lot of machinery into the test itself to make the individual
    callbacks reformat data internally. This mechanism required the
    data_specs field to be present. Weirdly, the theano functions
    never actually used any of the data, so their data_specs should
    have just been NullSpace anyway. IG deleted a lot of this useless
    code from these tests but there is still a lot of weird stuff here
    that he has not attempted to clean up.
    """

    rng = np.random.RandomState([2012, 11, 27, 9])

    batch_size = 5
    updates_per_batch = 4
    train_batches = 3
    num_features = 2

    # Synthesize dataset with a linear decision boundary
    w = rng.randn(num_features)

    def make_dataset(num_batches):
        m = num_batches*batch_size
        X = rng.randn(m, num_features)
        y = rng.randn(m, num_features)

        rval =  DenseDesignMatrix(X=X, y=y)

        rval.yaml_src = "" # suppress no yaml_src warning

        return rval

    train = make_dataset(train_batches)

    model = SoftmaxModel(num_features)

    unsup_counter = shared(0)
    grad_counter = shared(0)

    called = [False, False, False, False]

    class UnsupervisedCostWithFixedVars(Cost):

        def expr(self, model, data, unsup_aux_var=None, **kwargs):
            self.get_data_specs(model)[0].validate(data)
            X = data
            assert unsup_aux_var is unsup_counter
            called[0] = True
            return (model.P * X).sum()

        def get_gradients(self, model, data, unsup_aux_var=None, **kwargs):
            self.get_data_specs(model)[0].validate(data)
            assert unsup_aux_var is unsup_counter
            called[1] = True
            gradients, updates = Cost.get_gradients(self, model, data,
                    unsup_aux_var=unsup_aux_var)
            updates[grad_counter] = grad_counter + 1
            return gradients, updates

        def get_fixed_var_descr(self, model, data, **kwargs):
            data_specs = self.get_data_specs(model)
            data_specs[0].validate(data)
            rval = FixedVarDescr()
            rval.fixed_vars = {'unsup_aux_var': unsup_counter}

            # The input to function should be a flat, non-redundent tuple
            mapping = DataSpecsMapping(data_specs)
            data_tuple = mapping.flatten(data, return_tuple=True)
            theano_func = function([],
                    updates=[(unsup_counter, unsup_counter + 1)])
            def on_load(batch, mapping=mapping, theano_func=theano_func):
                return theano_func()
            rval.on_load_batch = [on_load]

            return rval

        def get_data_specs(self, model):
            return (model.get_input_space(), model.get_input_source())

    sup_counter = shared(0)

    class SupervisedCostWithFixedVars(Cost):

        supervised = True

        def expr(self, model, data, sup_aux_var=None, **kwargs):
            self.get_data_specs(model)[0].validate(data)
            X, Y = data
            assert sup_aux_var is sup_counter
            called[2] = True
            return (model.P * X * Y).sum()

        def get_gradients(self, model, data, sup_aux_var=None, **kwargs):
            self.get_data_specs(model)[0].validate(data)
            assert sup_aux_var is sup_counter
            called[3] = True
            return super(SupervisedCostWithFixedVars, self).get_gradients(
                    model=model, data=data, sup_aux_var=sup_aux_var)

        def get_fixed_var_descr(self, model, data):
            data_specs = self.get_data_specs(model)
            data_specs[0].validate(data)
            rval = FixedVarDescr()
            rval.fixed_vars = {'sup_aux_var': sup_counter}

            theano_func = function([], updates=[(sup_counter,
                sup_counter + 1)])
            def on_load(data):
                theano_func()
            rval.on_load_batch = [on_load]
            return rval

        def get_data_specs(self, model):
            space = CompositeSpace((model.get_input_space(),
                                   model.get_output_space()))
            source = (model.get_input_source(), model.get_target_source())
            return (space, source)

    cost = SumOfCosts(costs=[UnsupervisedCostWithFixedVars(),
                             SupervisedCostWithFixedVars()])

    algorithm = BGD(cost=cost, batch_size=batch_size,
            conjugate=1, line_search_mode='exhaustive',
            updates_per_batch=updates_per_batch)

    algorithm.setup(model=model, dataset=train)

    # Make sure all the right methods were used to compute the updates
    assert all(called)

    algorithm.train(dataset=train)

    # Make sure the load_batch callbacks were called the right amount of times
    assert unsup_counter.get_value() == train_batches
    assert sup_counter.get_value() == train_batches

    # Make sure the gradient updates were run the right amount of times
    assert grad_counter.get_value() == train_batches * updates_per_batch
