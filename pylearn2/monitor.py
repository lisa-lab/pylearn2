"""
The module defining the Monitor and MonitorChannel objects used for
tracking the changes in values of various quantities throughout training
"""
import time
from theano import function, shared
import copy
from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import number_aware_alphabetical_key

class Monitor(object):
    """
    A class for monitoring Models while they are being trained.

    A monitor object records the number of minibatches and number of examples
    the model has trained, as well as any number of "channels" that track
    quantities of interest (examples: the objective function, measures of
    hidden unit activity, reconstruction error, sum of squared second
    derivatives, average norm of the weight vectors,  etc.)
    """
    def __init__(self, model):
        """
        Makes a monitor for `model`. Assumes the model has not been
        trained at all yet.

        Parameters
        ----------
        model : object
            An object that implements the `Model` interface specified in
            `pylearn2.models`.
        """
        self.model = model
        self.channels = {}
        self._num_batches_seen = 0
        self._examples_seen = 0
        self._dataset = None
        self._dirty = True
        self._rng_seed = None
        self.names_to_del = []
        # Determine whether the model should use topological or vector form of
        # examples. If the model acts on a space with more than the batch index
        # and channel dimension, the model has topological dimensions, so the
        # topological view of the data should be used.
        self.topo = len(model.get_input_space().make_theano_batch().type.broadcastable) > 2
        self.require_label = False

    def set_dataset(self, dataset, mode, batch_size=None, num_batches=None):
        """
        Determines the data used to calculate the values of each channel.

        Parameters
        ----------
        dataset : object
            A `pylearn2.datasets.Dataset` object.
        mode : str or object, optional
            Iteration mode; see the docstring of the `iterator` method
            on `pylearn2.datasets.Dataset` for details.
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        """
        try:
            it = dataset.iterator(mode=mode, batch_size=batch_size,
                                  num_batches=num_batches,
                                  topo=self.topo,
                                  targets=self.require_label)
            # TODO: handle random seeds.
        except ValueError as exc:
            raise ValueError("invalid iteration parameters in "
                             "Monitor.set_dataset: " + str(exc))
        self._dataset = dataset
        self._iteration_mode = mode
        self._batch_size = batch_size
        self._num_batches = num_batches

    def __call__(self):
        """
        Runs the model on the monitoring dataset in order to add one
        data point to each of the channels.
        """
        if self._dirty:
            self.redo_theano()
        model = self.model
        d = self._dataset
        if d:
            if isinstance(d, basestring):
                d = yaml_parse.load(d)
                self._dataset = d
            myiterator = d.iterator(mode=self._iteration_mode,
                                    batch_size=self._batch_size,
                                    num_batches=self._num_batches,
                                    topo=self.topo,
                                    targets=self.require_label)
            self.begin_record_entry()
            count = 0
            for iteration, X in enumerate(myiterator):
                # make sure the iterator gave us the right size
                # the averaging code assumes all batches are the same size
                # assert X.shape[0] == self._batch_size
                self.run_prereqs(X)
                if self.require_label:
                    X, y = X
                    self.accum(X, y)
                else:
                    self.accum(X)
                count += 1

            # TODO: use logging infrastructure so that user can configure
            # formatting
            print "Monitoring step:"
            print "\tBatches seen: %d" % self._num_batches_seen
            print "\tExamples seen: %d" % self._examples_seen
            for channel_name in sorted(self.channels.keys(), key=number_aware_alphabetical_key):
                channel = self.channels[channel_name]
                channel.batch_record.append(self._num_batches_seen)
                channel.example_record.append(self._examples_seen)
                val = channel.val_shared.get_value(borrow=True)
                channel.val_record.append(val)
                # TODO: use logging infrastructure so that user can configure
                # formatting
                if abs(val) < 1e4:
                    val_str = str(val)
                else:
                    val_str = '%.3e' % val

                print "\t%s: %s" % (channel_name, val_str)

    def run_prereqs(self, X):
        for prereq in self.prereqs:
            prereq(X)


    def get_batches_seen(self):
        """ Returns the number of batches the model has learned on (assuming
        that the learning code has been calling Monitor.report_batch correctly)
        """
        return self._num_batches_seen

    def get_examples_seen(self):
        """ Returns the number of examples the model has learned on (assuming
        that the learning code has been calling Monitor.report_batch correctly)
        """
        return self._examples_seen

    def report_batch(self, num_examples):
        """ Call this whenever the model has learned on another batch of examples.
        Report how many examples were learned on. """
        self._examples_seen += num_examples
        self._num_batches_seen += 1

    def redo_theano(self):
        """
        Recompiles Theano functions used by this monitor.

        This is needed so that if new channels are added, Theano's
        optimizations make sure (to the extent that they can) that the new
        channels and old channels don't have any redundant calculations.

        It is also needed to regenerate Theano functions after pickling and
        unpickling, since Theano functions should not be pickled.
        """
        self._dirty = False

        self.prereqs = []
        for channel in self.channels.values():
            if channel.prereqs is not None:
                for prereq in channel.prereqs:
                    if prereq not in self.prereqs:
                        self.prereqs.append(prereq)

        init_names = dir(self)
        updates = {}
        for channel in self.channels.values():
            updates[channel.val_shared] = 0.0
        print "compiling begin_record_entry..."
        t1 = time.time()
        self.begin_record_entry = function(inputs=[], updates=updates)
        t2 = time.time()
        print "took " + str(t2 - t1) + " seconds"
        updates = {}
        givens = {}
        #Get the appropriate kind of theano variable to represent the data the model
        #acts on
        X = self.model.get_input_space().make_theano_batch(name = "monitoring_X")
        if self.require_label:
            Y = self.model.get_output_space().make_theano_batch(name = "monitoring_Y")

        print 'monitored channels: '+str(self.channels.keys())
        it = self.dataset.iterator(mode=self._iteration_mode,
                                   num_batches=self._num_batches,
                                   batch_size=self._batch_size)
        num_examples = float(it.num_examples)
        for channel in self.channels.values():
            if isinstance(channel.graph_input, (list, tuple)):
                givens[channel.graph_input[0]] = X
                givens[channel.graph_input[1]] = Y
            else:
                givens[channel.graph_input] = X
            val = channel.val * (X.shape[0] / num_examples)
            updates[channel.val_shared] = channel.val_shared + val
        print "compiling accum..."
        t1 = time.time()
        if self.require_label:
            self.accum = function([X, Y], givens=givens, updates=updates)
        else:
            self.accum = function([X], givens=givens, updates=updates)
        t2 = time.time()
        print "graph size: ",len(self.accum.maker.env.toposort())
        print "took "+str(t2-t1)+" seconds"
        final_names = dir(self)
        self.register_names_to_del([name for name in final_names
                                    if name not in init_names])

    def register_names_to_del(self, names):
        """
        Register names of fields that should be deleted before pickling.

        Parameters
        ----------
        names : list
            A list of attribute names as strings.
        """
        for name in names:
            if name not in self.names_to_del:
                self.names_to_del.append(name)

    def __getstate__(self):
        """
        In order to avoid pickling a copy of the dataset whenever a monitor
        is saved, the __getstate__ method replaces the dataset field with the
        dataset's yaml source. This is not a perfect solution because it won't
        work with job resuming, which would require saving the state of the
        dataset's random number generator.

        Like in the Model class, we also need to avoid saving any Theano
        functions, so we delete everything that can be regenerated with
        `redo_theano` by deleting the fields in `self.names_to_del`
        """
        temp = self._dataset
        if self._dataset and not isinstance(self._dataset, basestring):
            try:
                self._dataset = self._dataset.yaml_src
            except AttributeError:
                import warnings
                warnings.warn('Trained model saved without indicating yaml_src')
        d = copy.copy(self.__dict__)
        self._dataset = temp
        for name in self.names_to_del:
            if name in d:
                del d[name]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def add_channel(self, name, ipt, val, prereqs=None):
        """
        Asks the monitor to start tracking a new value.  Can be called even
        after the monitor is already in use.

        Parameters
        ----------
        name: str
            The display name in the monitor.
        ipt: tensor_like
            The symbolic tensor which should be clamped to the data.
        val: tensor_like
            The value (function of `ipt`) to be tracked.
        """

        if name in self.channels:
            raise ValueError("Tried to create the same channel twice (%s)" %
                             name)
        if isinstance(ipt, (list, tuple)):
            self.require_label = True
        self.channels[name] = MonitorChannel(ipt, val, name, prereqs)
        self._dirty = True

    @classmethod
    def get_monitor(cls, model):
        """
        Returns a model's monitor. If the model doesn't have a monitor yet,
        installs one and returns that.

        Parameters
        ----------
        model : object
            An object that implements the `Model` interface specified in
            `pylearn2.models`.
        """
        if hasattr(model, 'monitor'):
            rval = model.monitor
        else:
            rval = Monitor(model)
            model.monitor = rval
        return rval

    # TODO: find out if monitor.foo below are used anywhere, remove if not.
    @property
    def dataset(self):
        return self._dataset

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return self._num_batches


class MonitorChannel(object):
    """
    A class representing a specific quantity to be monitored.
    """
    def __init__(self, graph_input, val, name, prereqs=None):
        """
        Creates a channel for a quantity to be monitored.

        Parameters
        ----------
        graph_input : tensor_like
            The symbolic tensor which should be clamped to the data.
        val : tensor_like
            The value (symbolic function of `graph_input`) to be evaluated
            and recorded.
        name : str
            The display name in the monitor.
        prereqs: list of callables that take tensors
            each prereq must be called exactly once per each new
            batch of data before the channel value is computed
            if two channels provide a prereq with exactly the same
            id, that prereq will only be called once
        """
        self.prereqs = prereqs
        self.graph_input = graph_input
        self.val = val
        self.val_shared = shared(0.0, name + "_tracker")
        # Value of the desired quantity at measurement time.
        self.val_record = []
        # Number of batches seen at measurement time.
        self.batch_record = []
        # Number of examples seen at measurement time (batch sizes may
        # fluctuate).
        self.example_record = []

    def __getstate__(self):
        """ TODO:
                we need to figure out a good way of saving the other fields.
                in the current setup, since there's no good way of coordinating
                with the model/training algorithm, the theano based fields might
                be invalid after a repickle.
                This means we can't, for instance, resume a job with monitoring
                after a crash.
                For now, to make sure no one erroneously depends on these bad
                values, I exclude them from the pickle.
        """
        return {
            'example_record': self.example_record,
            'val_record': self.val_record
        }

    def __setstate__(self, d):
        self.__dict__.update(d)

