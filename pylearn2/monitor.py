"""TODO: module-level docstring."""
from theano import function, shared
import theano.tensor as T
import copy
from pylearn2.config import yaml_parse


class Monitor(object):
    """TODO: class-level docstring. What does this do?"""
    def __init__(self, model):
        """TODO: Document me."""
        self.model = model
        self.channels = {}
        self.batches_seen = 0
        self.examples_seen = 0
        self.dataset = None
        self.dirty = True
        self.names_to_del = []

    def set_dataset(self, dataset, batches, batch_size):
        """TODO: Document me."""
        self.dataset = dataset
        self.batches = batches
        self.batch_size = batch_size

    def __call__(self):
        if self.dirty:
            self.redo_theano()

        model = self.model
        d = self.dataset

        if d:
            if isinstance(d, str):
                d = yaml_parse.load(d)
                self.dataset = d

            s = d.get_stream_position()

            d.restart_stream()

            self.begin_record_entry()

            for i in xrange(self.batches):
                X = d.get_batch_design(self.batch_size)
                self.accum(X)
            self.finish_record_entry()
            d.set_stream_position(s)

    def finish_record_entry(self):
        """TODO: Document me."""
        print "Monitoring step:"
        print "\tBatches seen: %d" % self.batches_seen
        print "\tExamples seen: %d" % self.examples_seen
        for channel_name in self.channels:
            channel = self.channels[channel_name]
            channel.batch_record.append(self.batches_seen)
            channel.example_record.append(self.examples_seen)
            val = (channel.val_shared.get_value(borrow=False) /
                   float(self.batches))
            channel.val_record.append(val)
            print "\t%s: %s" % (channel_name, str(val))

    def redo_theano(self):
        """TODO: document me."""
        init_names = dir(self)
        updates = {}
        for channel in self.channels.values():
            updates[channel.val_shared] = 0.0
        self.begin_record_entry = function(inputs=[], updates=updates)
        updates = {}
        givens = {}
        X = T.matrix()
        for channel in self.channels.values():
            givens[channel.ipt] = X
            updates[channel.val_shared] = channel.val_shared + channel.val
        self.accum = function([X], givens=givens, updates=updates)
        final_names = dir(self)
        self.register_names_to_del([name for name in final_names
                                    if name not in init_names])

    def register_names_to_del(self, names):
        """TODO: document me."""
        for name in names:
            if name not in self.names_to_del:
                self.names_to_del.append(name)

    def __getstate__(self):
        """TODO: Document me (why specifically is this needed)"""
        temp = self.dataset
        if not isinstance(self.dataset, str):
            self.dataset = self.dataset.yaml_src
        d = copy.copy(self.__dict__)
        self.dataset = temp

        for name in self.names_to_del:
            if name in d:
                del d[name]

        return d

    def __setstate__(self, d):
        """TODO: is this necessary?"""
        self.__dict__.update(d)

    def add_channel(self, name, ipt, val):
        """TODO: Document me."""
        if name in self.channels:
            raise ValueError("Tried to create the same channel twice (%s)" %
                             name)
        self.channels[name] = Channel(ipt, val)
        self.dirty = True

    @classmethod
    def get_monitor(cls, model):
        """TODO: document me."""
        if hasattr(model, 'monitor'):
            rval = model.monitor
        else:
            rval = Monitor(model)
            model.monitor = rval
        return rval


class Channel(object):
    def __init__(self, ipt, val):
        self.ipt = ipt
        self.val = val
        self.val_shared = shared(0.0)
        self.batch_record = []
        self.example_record = []
        self.val_record = []
