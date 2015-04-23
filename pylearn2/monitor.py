"""
The module defining the Monitor and MonitorChannel objects used for
tracking the changes in values of various quantities throughout training
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import copy
import time
import warnings
import logging
import numpy as np
from theano.compat import six

from pylearn2.compat import OrderedDict
import theano.sparse
from theano import config
from theano import tensor as T
from theano.printing import var_descriptor

from pylearn2.config import yaml_parse
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import Space, CompositeSpace, NullSpace
from pylearn2.utils import function, sharedX, safe_zip, safe_izip
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.string_utils import number_aware_alphabetical_key
from pylearn2.utils.timing import log_timing

log = logging.getLogger(__name__)


class Monitor(object):
    """
    A class for monitoring Models while they are being trained.

    A monitor object records the number of minibatches and number of
    examples the model has trained, as well as any number of "channels"
    that track quantities of interest (examples: the objective
    function, measures of hidden unit activity, reconstruction error,
    sum of squared second derivatives, average norm of the weight
    vectors, etc.)

    Parameters
    ----------
    model : `pylearn2.models.model.Model`

    Attributes
    ----------
    on_channel_conflict : string
        `error` : this is a behavior when there is conlfict
            on creating a channel twice
        `copy_history` : this is a behavior when creating a
            new channel and transfering history of old_monitor
        `overwrite` : this is a behavior when creating a
            new channel without taking an account of old_monitor
    """

    def __init__(self, model):
        self.training_succeeded = False
        self.model = model
        self.channels = OrderedDict()
        self._num_batches_seen = 0
        self._examples_seen = 0
        self._epochs_seen = 0
        self._datasets = []
        self._iteration_mode = []
        self._batch_size = []
        self._num_batches = []
        self._dirty = True
        self._rng_seed = []
        self.names_to_del = ['theano_function_mode']
        self.t0 = time.time()
        self.theano_function_mode = None
        self.on_channel_conflict = 'error'

        # Initialize self._nested_data_specs, self._data_specs_mapping,
        # and self._flat_data_specs
        self._build_data_specs()

    def _build_data_specs(self):
        """
        Computes a nested data_specs for input and all channels

        Also computes the mapping to flatten it. This function is
        called from redo_theano.
        """
        # Ask the model what it needs
        m_space, m_source = self.model.get_monitoring_data_specs()
        input_spaces = [m_space]
        input_sources = [m_source]
        for channel in self.channels.values():
            space = channel.data_specs[0]
            assert isinstance(space, Space)
            input_spaces.append(space)
            input_sources.append(channel.data_specs[1])

        nested_space = CompositeSpace(input_spaces)
        nested_source = tuple(input_sources)

        self._nested_data_specs = (nested_space, nested_source)
        self._data_specs_mapping = DataSpecsMapping(self._nested_data_specs)

        flat_space = self._data_specs_mapping.flatten(nested_space,
                                                      return_tuple=True)
        flat_source = self._data_specs_mapping.flatten(nested_source,
                                                       return_tuple=True)
        self._flat_data_specs = (CompositeSpace(flat_space), flat_source)

    def set_theano_function_mode(self, mode):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        mode : theano.compile.Mode
            Theano functions for the monitoring channels will be
            compiled and run using this mode.
        """
        if self.theano_function_mode != mode:
            self._dirty = True
            self.theano_function_mode = mode

    def add_dataset(self, dataset, mode='sequential', batch_size=None,
                    num_batches=None, seed=None):
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
        seed : int, optional
            Optional. The seed to be used for random iteration modes.
        """
        # The user can ommit using lists if only one dataset is set
        if not isinstance(dataset, list):
            dataset = [dataset]
        if not isinstance(mode, list):
            mode = [mode]
        if not isinstance(batch_size, list):
            batch_size = [batch_size]
        if not isinstance(num_batches, list):
            num_batches = [num_batches]
        if seed is None:
            seed = [None] * len(dataset)
        if not isinstance(seed, list):
            seed = [seed]
        if len(mode) != len(dataset):
            raise ValueError("Received " + str(len(dataset)) +
                             " dataset but " + str(len(mode)) + " modes.")
        if any([len(l) != len(dataset) for l in [batch_size, seed]]):
            raise ValueError("make sure each dataset has its iteration " +
                             "batch size and number of batches.")
        for (d, m, b, n, sd) in safe_izip(dataset, mode, batch_size,
                                          num_batches, seed):
            try:
                it = d.iterator(mode=m,
                                batch_size=b,
                                num_batches=n,
                                data_specs=self._flat_data_specs,
                                return_tuple=True,
                                rng=sd)
            except ValueError as exc:
                reraise_as(ValueError("invalid iteration parameters in " +
                                      "Monitor.add_dataset: " + str(exc)))
            if it.stochastic:
                # Must be a seed, not a random number generator. If it were a
                # random number generator, different iterators using it would
                # update its state, so we would not get the same iterator
                # each time. Also, must not be None, because this makes the
                # iterator pick a seed based on the clock
                if sd is None:
                    raise TypeError("Monitor requires a seed when using " +
                                    "stochastic iteration modes.")
                if not isinstance(sd, (list, tuple, int)):
                    raise TypeError("Monitor requires a seed (not a random " +
                                    "number generator) when using " +
                                    "stochastic iteration modes.")
            else:
                # The iterator should catch this, but let's double-check
                assert sd is None

            if d not in self._datasets:
                self._datasets.append(d)
                self._iteration_mode.append(m)
                self._batch_size.append(b)
                self._num_batches.append(n)
                self._rng_seed.append(sd)

    def __call__(self):
        """
        Runs the model on the monitoring dataset in order to add one
        data point to each of the channels.
        """

        # If the channels have changed at all, we need to recompile the theano
        # functions used to compute them
        if self._dirty:
            self.redo_theano()

        datasets = self._datasets

        # Set all channels' val_shared to 0
        self.begin_record_entry()
        for d, i, b, n, a, sd, ne in safe_izip(datasets,
                                               self._iteration_mode,
                                               self._batch_size,
                                               self._num_batches,
                                               self.accum,
                                               self._rng_seed,
                                               self.num_examples):
            if isinstance(d, six.string_types):
                d = yaml_parse.load(d)
                raise NotImplementedError()

            # need to put d back into self._datasets
            myiterator = d.iterator(mode=i,
                                    batch_size=b,
                                    num_batches=n,
                                    data_specs=self._flat_data_specs,
                                    return_tuple=True,
                                    rng=sd)

            # If self._flat_data_specs is empty, no channel needs data,
            # so we do not need to call the iterator in order to average
            # the monitored values across different batches, we only
            # have to call them once.
            if len(self._flat_data_specs[1]) == 0:
                X = ()
                self.run_prereqs(X, d)
                a(*X)

            else:
                actual_ne = 0
                for X in myiterator:
                    # X is a flat (not nested) tuple
                    self.run_prereqs(X, d)
                    a(*X)
                    actual_ne += self._flat_data_specs[0].np_batch_size(X)
                # end for X
                if actual_ne != ne:
                    raise RuntimeError("At compile time, your iterator said "
                                       "it had %d examples total, but at "
                                       "runtime it gave us %d." %
                                       (ne, actual_ne))
        # end for d

        log.info("Monitoring step:")
        log.info("\tEpochs seen: %d" % self._epochs_seen)
        log.info("\tBatches seen: %d" % self._num_batches_seen)
        log.info("\tExamples seen: %d" % self._examples_seen)
        t = time.time() - self.t0
        for channel_name in sorted(self.channels.keys(),
                                   key=number_aware_alphabetical_key):
            channel = self.channels[channel_name]
            channel.time_record.append(t)
            channel.batch_record.append(self._num_batches_seen)
            channel.example_record.append(self._examples_seen)
            channel.epoch_record.append(self._epochs_seen)
            val = channel.val_shared.get_value()
            channel.val_record.append(val)
            # TODO: use logging infrastructure so that user can configure
            # formatting
            if abs(val) < 1e4:
                val_str = str(val)
            else:
                val_str = '%.3e' % val

            log.info("\t%s: %s" % (channel_name, val_str))

    def run_prereqs(self, data, dataset):
        """
        Runs all "prerequistie functions" on a batch of data. Always
        called right before computing the monitoring channels on that
        batch.

        Parameters
        ----------
        data : tuple or Variable
            a member of the Space used as input to the monitoring
            functions
        dataset : Dataset
            the Dataset the data was drawn from
        """
        if dataset not in self.prereqs:
            return
        for prereq in self.prereqs[dataset]:
            prereq(*data)

    def get_batches_seen(self):
        """
        Returns the number of batches the model has learned on
        (assuming that the learning code has been calling
        Monitor.report_batch correctly).
        """
        return self._num_batches_seen

    def get_epochs_seen(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        epochs_seen : int
            The number of epochs the model has been trained on.
            One "epoch" is one pass through Dataset.iterator.
        """
        return self._epochs_seen

    def get_examples_seen(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        examples_seen : int
            The number of examples the model has learned on (assuming
            that the learning code has been calling Monitor.report_batch
            correctly)
        """
        return self._examples_seen

    def report_batch(self, num_examples):
        """
        Call this whenever the model has learned on another batch of
        examples. Report how many examples were learned on.

        Parameters
        ----------
        num_examples : int
            The number of examples learned on in this minibatch.
        """
        self._examples_seen += num_examples
        self._num_batches_seen += 1

    def report_epoch(self):
        """
        Call this whenever the model has completed another "epoch" of
        learning. We regard one pass through Dataset.iterator as one
        epoch.
        """
        self._epochs_seen += 1

    def redo_theano(self):
        """
        Recompiles Theano functions used by this monitor.

        This is called any time we need to evaluate the channels and
        the channel definitions have changed since last we called it,
        or if the theano functions are unavailable for any other reason
        (first time they are needed after construction or
        deserialization, etc.)

        All channels are compiled as part of the same theano function
        so that the theano optimizations can eliminate subexpressions
        that are shared between multiple channels.
        """
        self._dirty = False

        # Recompute the data specs, since the channels may have changed.
        self._build_data_specs()

        init_names = dir(self)
        self.prereqs = OrderedDict()
        for channel in self.channels.values():
            if channel.prereqs is not None:
                dataset = channel.dataset
                if dataset not in self.prereqs:
                    self.prereqs[dataset] = []
                prereqs = self.prereqs[dataset]
                for prereq in channel.prereqs:
                    if prereq not in prereqs:
                        prereqs.append(prereq)

        updates = OrderedDict()
        for channel in self.channels.values():
            updates[channel.val_shared] = np.cast[config.floatX](0.0)
        with log_timing(log, "compiling begin_record_entry"):
            self.begin_record_entry = function(
                inputs=[],
                updates=updates,
                mode=self.theano_function_mode,
                name='Monitor.begin_record_entry'
            )
        updates = OrderedDict()
        givens = OrderedDict()
        # Get the appropriate kind of theano variable to represent the data
        # the model acts on
        batch_names = ['monitoring_%s' % s for s in self._flat_data_specs[1]]
        theano_args = self._flat_data_specs[0].make_theano_batch(batch_names)

        # Get a symbolic expression of the batch size
        # We do it here, rather than for each channel, because channels with an
        # empty data_specs do not use data, and are unable to extract the batch
        # size. The case where the whole data specs is empty is not supported.
        batch_size = self._flat_data_specs[0].batch_size(theano_args)

        # Also get a nested representation, for joint iteration
        # with each of channel.graph_input
        nested_theano_args = self._data_specs_mapping.nest(theano_args)
        if not isinstance(nested_theano_args, tuple):
            nested_theano_args = (nested_theano_args,)
        assert len(nested_theano_args) == (len(self.channels) + 1)

        log.info('Monitored channels: ')
        for key in sorted(self.channels.keys()):
            mode = self.theano_function_mode
            if mode is not None and hasattr(mode, 'record'):
                mode.record.handle_line('compiling monitor including ' +
                                        'channel ' + key + '\n')
            log.info('\t%s' % key)
        it = []
        for d, i, n, b in safe_izip(self._datasets, self._iteration_mode,
                                    self._num_batches, self._batch_size):
            it.append(d.iterator(mode=i, num_batches=n, batch_size=b,
                                 data_specs=self._flat_data_specs,
                                 return_tuple=True))
        self.num_examples = [i.num_examples for i in it]
        givens = [OrderedDict() for d in self._datasets]
        updates = [OrderedDict() for d in self._datasets]
        for i, channel in enumerate(self.channels.values()):
            index = self._datasets.index(channel.dataset)
            d = self._datasets[index]
            g = givens[index]
            cur_num_examples = self.num_examples[index]
            u = updates[index]

            # Flatten channel.graph_input and the appropriate part of
            # nested_theano_args, to iterate jointly over them.
            c_mapping = DataSpecsMapping(channel.data_specs)
            channel_inputs = c_mapping.flatten(channel.graph_input,
                                               return_tuple=True)
            inputs = c_mapping.flatten(nested_theano_args[i + 1],
                                       return_tuple=True)

            for (channel_X, X) in safe_izip(channel_inputs, inputs):
                assert channel_X not in g or g[channel_X] is X
                assert channel_X.type == X.type, (channel_X.type, X.type)
                g[channel_X] = X

            if batch_size == 0:
                # No channel does need any data, so there is not need to
                # average results, and we will call the accum functions only
                # once.
                # TODO: better handling of channels not needing data when
                # some other channels need data.
                assert len(self._flat_data_specs[1]) == 0
                val = channel.val
            else:
                if n == 0:
                    raise ValueError("Iterating over 0 examples results in " +
                                     "divide by 0")
                val = T.cast(channel.val * T.cast(batch_size, 'float64')
                             / cur_num_examples, config.floatX)
            u[channel.val_shared] = channel.val_shared + val

        with log_timing(log, "Compiling accum"):
            # Check type of update expressions
            for up in updates:
                for key in up:
                    if key.dtype != up[key].dtype:
                        raise TypeError('Monitoring channel shared variable ' +
                                        key.name + ' has dtype ' + key.dtype +
                                        ' but is driven by an expression ' +
                                        'with type ' + up[key].dtype)

            self.accum = []
            for idx, packed in enumerate(safe_izip(givens, updates)):
                g, u = packed
                mode = self.theano_function_mode
                if mode is not None and hasattr(mode, 'record'):
                    for elem in g:
                        mode.record.handle_line('g key ' +
                                                var_descriptor(elem) + '\n')
                        mode.record.handle_line('g val ' +
                                                var_descriptor(g[elem]) + '\n')
                    for elem in u:
                        mode.record.handle_line('u key ' +
                                                var_descriptor(elem) + '\n')
                        mode.record.handle_line('u val ' +
                                                var_descriptor(u[elem]) + '\n')
                function_name = 'Monitor.accum[%d]' % idx
                if mode is not None and hasattr(mode, 'record'):
                    mode.record.handle_line('compiling supervised accum\n')
                # Some channels may not depend on the data, ie, they might just
                # monitor the model parameters, or some shared variable updated
                # by the training algorithm, so we need to ignore the unused
                # input error
                self.accum.append(function(theano_args,
                                           givens=g,
                                           updates=u,
                                           mode=self.theano_function_mode,
                                           name=function_name))
            for a in self.accum:
                if mode is not None and hasattr(mode, 'record'):
                    for elem in a.maker.fgraph.outputs:
                        mode.record.handle_line('accum output ' +
                                                var_descriptor(elem) + '\n')
                log.info("graph size: %d" % len(a.maker.fgraph.toposort()))
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
        In order to avoid pickling a copy of the dataset whenever a
        monitor is saved, the __getstate__ method replaces the dataset
        field with the dataset's yaml source. This is not a perfect
        solution because it won't work with job resuming, which would
        require saving the state of the dataset's random number
        generator.

        Like in the Model class, we also need to avoid saving any
        Theano functions, so we delete everything that can be
        regenerated with `redo_theano` by deleting the fields in
        `self.names_to_del`
        """

        # Patch old pickled monitors
        if not hasattr(self, '_datasets'):
            self._datasets = [self._dataset]
            del self._dataset

        temp = self._datasets

        if self._datasets:
            self._datasets = []
            for dataset in temp:
                if isinstance(dataset, six.string_types):
                    self._datasets.append(dataset)
                else:
                    try:
                        self._datasets.append(dataset.yaml_src)
                    except AttributeError:
                        warnings.warn('Trained model saved without ' +
                                      'indicating yaml_src')
        d = copy.copy(self.__dict__)
        self._datasets = temp
        for name in self.names_to_del:
            if name in d:
                del d[name]

        return d

    def __setstate__(self, d):
        """
        Sets the object to have the state described by `d`.

        Parameters
        ----------
        d : dict
            A dictionary mapping string names of fields to values for
            these fields.
        """
        # patch old pkl files
        if '_dataset' in d:
            d['_datasets'] = [d['_dataset']]
            del d['_dataset']

        self.__dict__.update(d)

    def add_channel(self, name, ipt, val, dataset=None, prereqs=None,
                    data_specs=None):
        """
        Asks the monitor to start tracking a new value.  Can be called
        even after the monitor is already in use.

        Parameters
        ----------
        name : str
            The display name in the monitor.
        ipt : tensor_like
            The symbolic tensor which should be clamped to the data.
            (or a list/tuple containing symbolic tensors, following the
            data_specs)
        val : tensor_like
            The value (function of `ipt`) to be tracked.
        dataset : pylearn2.datasets.Dataset
            Which dataset to compute this channel on
        prereqs : list of callables that take a list of numpy tensors
            Each prereq must be called exactly once per each new batch
            of data drawn *from dataset* before the channel value is
            computed if two channels provide a prereq with exactly the
            same id, that prereq will only be called once
        data_specs : (space, source) pair
            Identifies the order, format and semantics of ipt
        """
        if six.PY3:
            numeric = (float, int)
        else:
            numeric = (float, int, long)  # noqa

        if isinstance(val, numeric):
            val = np.cast[theano.config.floatX](val)

        val = T.as_tensor_variable(val)

        if data_specs is None:
            warnings.warn("parameter 'data_specs' should be provided when " +
                          "calling add_channel. We will build a default one.",
                          stacklevel=2)
            if isinstance(ipt, list):
                ipt = tuple(ipt)
            if ipt is not None and not isinstance(ipt, tuple):
                ipt = (ipt,)

            if ipt is None:
                data_specs = (NullSpace(), '')
            elif len(ipt) == 0:
                data_specs = (CompositeSpace([]), ())
            elif hasattr(dataset, 'get_data_specs'):
                dataset_space, dataset_source = dataset.get_data_specs()
                if (len(ipt) == 1 and
                        dataset_source is not None and
                        (not isinstance(dataset_source, tuple) or
                            len(dataset_source) == 1) and
                        'features' in dataset_source):
                    data_specs = (dataset_space, dataset_source)
                elif (len(ipt) == 2 and
                        dataset_source == ('features', 'targets')):
                    data_specs = (dataset_space, dataset_source)
                else:
                    raise ValueError("Cannot infer default data_specs for " +
                                     "the following input points and " +
                                     "dataset: ipt = %s, dataset = %s"
                                     % (ipt, dataset))

        data_specs[0].validate(ipt)

        mapping = DataSpecsMapping(data_specs)
        flat_ipt = mapping.flatten(ipt)
        if not isinstance(flat_ipt, tuple):
            flat_ipt = (flat_ipt,)
        inputs = theano.gof.graph.inputs([val])
        for elem in inputs:
            if not hasattr(elem, 'get_value') and \
               not isinstance(elem, theano.gof.graph.Constant):
                if elem not in flat_ipt:
                    raise ValueError("Unspecified input: " + str(elem) +
                                     ". This may be due to an incorrect " +
                                     "implementation of a cost's " +
                                     "get_data_specs() method, or of a " +
                                     "model's get_monitoring_data_specs() " +
                                     "method.")

        mode = self.theano_function_mode
        if mode is not None and hasattr(mode, 'record'):
            mode.record.handle_line('Adding monitor channel '+name+'\n')
            assert isinstance(flat_ipt, tuple)
            if len(flat_ipt) != 1:
                for elem in flat_ipt:
                    mode.record.handle_line('Includes input var ' +
                                            var_descriptor(elem) + '\n')
            else:
                mode.record.handle_line(name + ' input var is ' +
                                        var_descriptor(flat_ipt[0]) + '\n')
            mode.record.handle_line('channel ' + name + ' is ' +
                                    var_descriptor(val) + '\n')

        if dataset is None:
            if len(self._datasets) == 1:
                dataset = self._datasets[0]
            elif len(self._datasets) == 0:
                raise ValueError(_err_no_data)
            else:
                raise ValueError(_err_ambig_data)

        try:
            self._datasets.index(dataset)
        except ValueError:
            reraise_as(ValueError("The dataset specified is not one of the " +
                                  "monitor's datasets"))

        if ((self.on_channel_conflict not in
             ('error', 'copy_history', 'overwrite'))):
            raise ValueError("on_channel_conflict should be either 'error'" +
                             "'copy_history', or 'overwrite'")

        if name in self.channels and self.on_channel_conflict == 'error':
            raise ValueError("Tried to create the same channel twice (%s)" %
                             name)
        elif ((name in self.channels and
               self.on_channel_conflict == 'copy_history')):
            self.channels[name] = MonitorChannel(ipt, val, name, data_specs,
                                                 dataset, prereqs,
                                                 self.channels[name])
        elif ((name not in self.channels or
               self.on_channel_conflict == 'overwrite')):
            self.channels[name] = MonitorChannel(ipt, val, name, data_specs,
                                                 dataset, prereqs)
        self._dirty = True

    def _sanity_check(self):
        """
        Sometimes we serialize models and then load them somewhere else
        but still try to use their Monitor, and the Monitor is in a
        mangled state. I've added some calls to _sanity_check to try to
        catch when that happens. Not sure what to do for a long term
        fix. I think it requires making theano graphs serializable
        first.
        """
        for name in self.channels:
            channel = self.channels[name]
            assert hasattr(channel, 'prereqs')

    @classmethod
    def get_monitor(cls, model):
        """
        Returns a model's monitor. If the model doesn't have a monitor
        yet, installs one and returns that.

        Parameters
        ----------
        model : object
            An object that implements the `Model` interface specified
            in `pylearn2.models`.
        """

        if hasattr(model, 'monitor'):
            rval = model.monitor
            rval._sanity_check()
        else:
            rval = Monitor(model)
            model.monitor = rval

        return rval

    # TODO: find out if this method is used anywhere, remove if not.
    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        batch_size : int
            The size of the batches used for monitoring
        """
        return self._batch_size

    # TODO: find out if this method is used anywhere, remove if not.
    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        num_batches : int
            The number of batches used for monitoring
        """
        return self._num_batches

    def setup(self, dataset, cost, batch_size, num_batches=None,
              extra_costs=None, mode='sequential', obj_prereqs=None,
              cost_monitoring_args=None):
        """
        Sets up the monitor for a cost minimization problem.
        Adds channels defined by both the model and the cost for
        the specified dataset(s), as well as a channel called
        'objective' defined by the costs' __call__ method.

        Parameters
        ----------
        dataset : pylearn2.datasets.Dataset
            Dataset or dictionary mapping string names to Datasets.
            If string names are used, then for every dataset, each
            channel defined by the model or cost will be replicated
            with that dataset's name followed by an underscore as the
            prefix. For example, if your cost defines a channel called
            'misclass', and datasets is
            {'train' : train_dataset, 'valid' : valid_dataset},
            you will get channels called 'train_misclass' and
            'valid_misclass'.
        cost : pylearn2.costs.Cost
            The cost being optimized by training. The value of the cost
            will appear as the `objective` channel. Its
            `get_monitoring_channels` method will also be used to
            supply other channels.
        extra_costs : OrderedDict, optional
            A dictionary mapping channel names to Cost objects.
            Their value will appear as the specified channel name.
            They will also provide more monitoring channels via their
            `get_monitoring_channels` method.
        obj_prereqs : None, or list of functions
            Functions to pass as prerequisites to the `objective` channel.
        cost_monitoring_args : dict
            Dictionary of kwargs that will be passed to
            `cost.get_monitoring_channels()`
            (but not for the extra_costs).
        """

        if dataset is None:
            return
        if isinstance(dataset, Dataset):
            dataset = {'': dataset}
        else:
            assert isinstance(dataset, dict)
            assert all(isinstance(key, str) for key in dataset)
            assert all(isinstance(dataset[key], Dataset) for key in dataset)

        if extra_costs is None:
            costs = {}
        else:
            assert isinstance(extra_costs, (OrderedDict, dict))
            costs = extra_costs
        assert '' not in costs
        costs[''] = cost

        if cost_monitoring_args is None:
            cost_monitoring_args = {}

        model = self.model

        # Build a composite data_specs containing the specs for all costs,
        # then the specs of the model
        cost_names = sorted(costs.keys())
        spaces = []
        sources = []
        for c in cost_names:
            c_space, c_source = costs[c].get_data_specs(model)
            spaces.append(c_space)
            sources.append(c_source)

        # Ask the model for the data_specs needed
        m_space, m_source = model.get_monitoring_data_specs()
        spaces.append(m_space)
        sources.append(m_source)

        nested_space = CompositeSpace(spaces)
        nested_sources = tuple(sources)

        # Flatten this data_specs, so we build only one symbolic Theano
        # variable for each of the unique (space, source) pairs.
        mapping = DataSpecsMapping((nested_space, nested_sources))
        space_tuple = mapping.flatten(nested_space, return_tuple=True)
        source_tuple = mapping.flatten(nested_sources, return_tuple=True)
        ipt = tuple(space.make_theano_batch(name='monitor_%s' % source,
                                            batch_size=None)
                    for (space, source) in safe_zip(space_tuple, source_tuple))

        # Build a nested tuple from ipt, to dispatch the appropriate parts
        # of the ipt batch to each cost
        nested_ipt = mapping.nest(ipt)

        custom_channels = {}
        for i, cost_name in enumerate(cost_names):
            if cost_name == '':
                prefix = ''
            else:
                prefix = cost_name + '_'
            cost = costs[cost_name]
            cost_ipt = nested_ipt[i]
            raw_channels = cost.get_monitoring_channels(model, cost_ipt)
            channels = {}
            for name in raw_channels:
                # We need three things: the value itself (raw_channels[name]),
                # the input variables (cost_ipt), and the data_specs for
                # these input variables ((spaces[i], sources[i]))
                channels[prefix + name] = (raw_channels[name],
                                           cost_ipt,
                                           (spaces[i], sources[i]))
            custom_channels.update(channels)

        # Use the last inputs from nested_ipt for the model
        model_channels = model.get_monitoring_channels(nested_ipt[-1])
        channels = {}
        for name in model_channels:
            # Note: some code used to consider that model_channels[name]
            # could be a a (channel, prereqs) pair, this is not supported.
            channels[name] = (model_channels[name],
                              nested_ipt[-1],
                              (spaces[-1], sources[-1]))
        custom_channels.update(channels)

        if is_stochastic(mode):
            seed = [[2013, 2, 22]]
        else:
            seed = None

        for dataset_name in dataset:
            cur_dataset = dataset[dataset_name]
            self.add_dataset(dataset=cur_dataset,
                             mode=mode,
                             batch_size=batch_size,
                             num_batches=num_batches,
                             seed=seed)
            if dataset_name == '':
                dprefix = ''
            else:
                dprefix = dataset_name + '_'
            # These channel name 'objective' must not vary, since callbacks
            # that respond to the values in the monitor use the name to find
            # it.
            for i, cost_name in enumerate(cost_names):
                cost = costs[cost_name]
                cost_ipt = nested_ipt[i]
                cost_value = cost.expr(model, cost_ipt)
                if cost_value is not None:
                    if cost_name == '':
                        name = dprefix + 'objective'
                        prereqs = obj_prereqs
                    else:
                        name = dprefix + cost_name
                        prereqs = None

                    cost.get_data_specs(model)[0].validate(cost_ipt)
                    self.add_channel(name=name,
                                     ipt=cost_ipt,
                                     val=cost_value,
                                     data_specs=cost.get_data_specs(model),
                                     dataset=cur_dataset,
                                     prereqs=prereqs)

            for key in custom_channels:
                val, ipt, data_specs = custom_channels[key]
                data_specs[0].validate(ipt)
                self.add_channel(name=dprefix + key,
                                 ipt=ipt,
                                 val=val,
                                 data_specs=data_specs,
                                 dataset=cur_dataset)


class MonitorChannel(object):
    """
    A class representing a specific quantity to be monitored.

    Parameters
    ----------
    graph_input : tensor_like
        The symbolic tensor which should be clamped to the data.
    val : tensor_like
        The value (symbolic function of `graph_input`) to be evaluated
        and recorded.
    name : str
        The display name in the monitor.
    data_specs : (space, source) pair
        Identifies the order, format and semantics of graph_input
    prereqs : list of callables
        Callables that take numpy tensors each prereq must be called
        exactly once per each new batch of data before the channel
        value is computed if two channels provide a prereq with exactly
        the same id, that prereq will only be called once
    old_channel : MonitorChannel
        MonitorChannel of old monitor, if not None, records of
        MonitorChannel will be initialized with records of old channel.
        When initializing the channel, the last value will be excluded,
        since it will be instantly recomputed by the next launch.
    """

    def __init__(self, graph_input, val, name, data_specs, dataset,
                 prereqs=None, old_channel=None):
        self.name = name
        self.prereqs = prereqs
        self.graph_input = graph_input
        self.data_specs = data_specs
        if isinstance(val, float):
            val = T.constant(np.cast[config.floatX](val))
        self.val = val
        self.val_shared = sharedX(0.0, name + "_tracker")
        assert self.val_shared.dtype == config.floatX, \
            "expected %s, got %s" % (config.floatX, self.val_shared.dtype)
        if not hasattr(val, 'dtype'):
            raise TypeError('Monitor channel ' + name + ' has value of type ' +
                            str(type(val)))
        if val.dtype != self.val_shared.dtype:
            raise ValueError('monitor channels are expected to have dtype ' +
                             str(self.val_shared.dtype) + ' but "' + name +
                             '" has dtype ' + str(val.dtype))
        if val.ndim != 0:
            raise ValueError('monitor channels are supposed to have zero ' +
                             'dimensions but "' + name + '" has ' +
                             str(val.ndim))
        # Dataset monitored by this channel
        self.dataset = dataset
        if old_channel is not None:
            # Value of the desired quantity at measurement time.
            self.val_record = old_channel.val_record[:-1]
            # Number of batches seen at measurement time.
            self.batch_record = old_channel.batch_record[:-1]
            # Number of examples seen at measurement time (batch sizes may
            # fluctuate).
            self.example_record = old_channel.example_record[:-1]
            self.epoch_record = old_channel.epoch_record[:-1]
            self.time_record = old_channel.time_record[:-1]
        else:
            # Value of the desired quantity at measurement time.
            self.val_record = []
            # Number of batches seen at measurement time.
            self.batch_record = []
            # Number of examples seen at measurement time (batch sizes may
            # fluctuate).
            self.example_record = []
            self.epoch_record = []
            self.time_record = []

    def __str__(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        s : str
            A reasonably human-readable string representation of the object.
        """
        try:
            graph_input_str = str(self.graph_input)
        except Exception:
            graph_input_str = '<bad graph input>'

        try:
            val_str = str(self.val)
        except Exception:
            val_str = '<bad val>'

        try:
            name_str = str(self.name)
        except Exception:
            name_str = '<bad name>'

        try:
            prereqs_str = str(self.prereqs)
        except Exception:
            prereqs_str = '<bad prereqs>'

        return "MonitorChannel(%s,%s,%s,%s)" % (graph_input_str,
                                                val_str,
                                                name_str,
                                                prereqs_str)

    def __getstate__(self):
        """
        .. todo::

            WRITEME

        Returns
        -------
        d : dict
            A dictionary mapping the string names of the fields of the class
            to values appropriate for pickling.
        """
        # We need to figure out a good way of saving the other fields. In the
        # current setup, since there's no good way of coordinating with the
        # model/training algorithm, the theano based fields might be invalid
        # after a repickle. This means we can't, for instance, resume a job
        # with monitoring after a crash. For now, to make sure no one
        # erroneously depends on these bad values, I exclude them from the
        # pickle.

        if hasattr(self, 'val'):
            doc = get_monitor_doc(self.val)
        else:
            # Hack to deal with Theano expressions not being serializable.
            # If this is channel that has been serialized and then
            # deserialized, the expression is gone, but we should have
            # stored the doc
            if hasattr(self, "doc"):
                doc = self.doc
            else:
                # Support pickle files that are older than the doc system
                doc = None

        return {
            'doc': doc,
            'example_record': self.example_record,
            'batch_record': self.batch_record,
            'time_record': self.time_record,
            'epoch_record': self.epoch_record,
            'val_record': self.val_record
        }

    def __setstate__(self, d):
        """
        Sets the object to have the state described by `d`.

        Parameters
        ----------
        d : dict
            A dictionary mapping string names of fields to values for
            these fields.
        """
        self.__dict__.update(d)
        if 'batch_record' not in d:
            self.batch_record = [None] * len(self.val_record)
        # Patch old pickle files that don't have the "epoch_record" field
        if 'epoch_record' not in d:
            # This is not necessarily correct but it is in the most common use
            # case where you don't add monitoring channels over time.
            self.epoch_record = range(len(self.val_record))
        if 'time_record' not in d:
            self.time_record = [None] * len(self.val_record)


def push_monitor(model, name, transfer_experience=False,
                 save_records=False):
    """
    When you load a model in a yaml file and you want to store its
    old monitor under a different name and start a new monitor, wrap
    the model in this function call.


    Parameters
    ----------
    model : pylearn2.models.model.Model
        The model you loaded
    name : str
        Will save the old monitor to model.name
    transfer_experience : bool
        If True, the new monitor will start with its epochs seen,
        batches seen, and examples seen set to where the old monitor
        left off. This is nice for stitching together learning curves
        across multiple stages of learning.
    save_records : bool
        If True, val_record, batch_record, example_record, epoch_record,
        and time_record of the new monitor will be initialzed with the
        records of old monitor.

    Returns
    -------
    model : WRITEME
        Returns the model itself so you can use an !obj:push_monitor
        call as the definition of a model in a YAML file.
    """

    assert hasattr(model, 'monitor')
    old_monitor = model.monitor
    setattr(model, name, old_monitor)
    del model.monitor

    if transfer_experience:
        monitor = Monitor.get_monitor(model)
        assert monitor is not old_monitor
        monitor._num_batches_seen = old_monitor._num_batches_seen
        monitor._examples_seen = old_monitor._examples_seen
        monitor._epochs_seen = old_monitor._epochs_seen
        if save_records:
            monitor.on_channel_conflict = 'copy_history'
            monitor.channels = copy.copy(old_monitor.channels)
            for key, value in list(monitor.channels.items()):
                value.prereqs = None

    return model


def read_channel(model, channel_name, monitor_name='monitor'):
    """
    Returns the last value recorded in a channel.

    Parameters
    ----------
    model : Model
        The model to read the channel from
    channel_name : str
        The name of the channel to read from
    monitor_name : str, optional
        The name of the Monitor to read from
        (In case you want to read from an old Monitor moved by
        `push_monitor`)

    Returns
    -------
    value : float
        The last value recorded in this monitoring channel
    """
    return getattr(model, monitor_name).channels[channel_name].val_record[-1]


def get_channel(model, dataset, channel, cost, batch_size):
    """
    Make a temporary monitor and return the value of a channel in it.

    Parameters
    ----------
    model : pylearn2.models.model.Model
        Will evaluate the channel for this Model.
    dataset : pylearn2.datasets.Dataset
        The Dataset to run on
    channel : str
        A string identifying the channel name to evaluate
    cost : pylearn2.costs.Cost
        The Cost to setup for monitoring
    batch_size : int
        The size of the batch to use when running the monitor

    Returns
    -------
    value : WRITEME
        The value of the requested channel.

    Notes
    -----
    This doesn't modify the model (unless some of the channel prereqs do).
    In particular, it does not change model.monitor.
    """
    monitor = Monitor(model)
    monitor.setup(dataset=dataset, cost=cost, batch_size=batch_size)
    monitor()
    channels = monitor.channels
    channel = channels[channel]
    val_record = channel.val_record
    value, = val_record
    return value


def get_monitor_doc(var):
    """
    Returns the __doc__ field of var or None. This field is used on
    theano Variables to document the meaning of monitor channels.

    Parameters
    ----------
    var : theano.gof.Variable
        The variable to get the documentation of

    Returns
    -------
    doc : str or None
        var.__doc__ if var has an instance-level doc, otherwise None
    """

    doc = None

    if var.__doc__ is not var.__class__.__doc__:
        doc = var.__doc__

    return doc

_err_no_data = "You tried to add a channel to a Monitor that has no dataset."
_err_ambig_data = ("You added a channel to a Monitor that has multiple " +
                   "datasets, and did not specify which dataset to use it " +
                   "with.")
