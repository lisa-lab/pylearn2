from theano import function, shared
import theano.tensor as T
import copy
from pylearn2.config import yaml_parse

class Monitor:

    def __init__(self, model):
        self.model = model
        self.channels = {}
        self.batches_seen = 0
        self.examples_seen = 0
        self.dataset = None
        self.dirty = True


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d['dataset'] = self.dataset.yaml_src
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.dataset = yaml_parse.load(d['dataset'])


    def set_dataset(self, dataset, batches, batch_size):
        self.dataset = dataset
        self.batches = batches
        self.batch_size = batch_size

    def __call__(self):
        if self.dirty:
            self.redo_theano()

        model = self.model
        d = self.dataset

        if d:
            s = d.get_stream_position()

            d.restart_stream()

            self.begin_record_entry()

            for i in xrange(self.batches):
                X = d.get_batch_design(self.batch_size)
                self.accum(X)

            self.finish_record_entry()


            d.set_stream_position(s)
        #
    #

    def finish_record_entry(self):
        print "Monitoring step:"
        print "\tBatches seen: "+str(self.batches_seen)
        print "\tExamples seen: "+str(self.examples_seen)
        for channel_name in self.channels:
            channel = self.channels[channel_name]
            channel.batch_record.append(self.batches_seen)
            channel.example_record.append(self.examples_seen)
            val = channel.val_shared.get_value(borrow=False) / float(self.batches)
            channel.val_record.append(val)
            print "\t",channel_name,": ",val

    def redo_theano(self):

        updates = {}

        for channel in self.channels.values():
            updates[channel.val_shared] = 0.0

        self.begin_record_entry = function(inputs = [], updates = updates)


        updates = {}
        givens  = {}

        X = T.matrix()

        for channel in self.channels.values():
            givens[channel.ipt] = X
            updates[channel.val_shared] = channel.val_shared + channel.val

        self.accum = function([X],givens = givens, updates = updates)



    def add_channel(self, name, ipt, val):
        if name in self.channels:
            raise ValueError("Tried to create the same channel, ",name," twice.")
        self.channels[name] = Channel(ipt, val)
        self.dirty = True

    @classmethod
    def get_monitor(cls, model):
        if hasattr(model,'monitor'):
            rval =  model.monitor
        else:
            rval = Monitor(model)
            model.monitor = rval
        #
        return rval
    #

class Channel:
    def __init__(self, ipt, val):
        self.ipt = ipt
        self.val = val
        self.val_shared = shared(0.0)
        self.batch_record = []
        self.example_record = []
        self.val_record = []


