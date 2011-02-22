"""
Several utilities for experimenting upon utlc datasets
"""
# Standard library imports
import os, sys
from itertools import izip

# Third-party imports
import numpy
import theano
from theano import tensor
from pylearn.datasets.utlc import load_ndarray_dataset

##################################################
# Shortcuts and auxiliary functions
##################################################

floatX = theano.config.floatX

def subdict(d, keys):
    """ Create a subdictionary of d with the keys in keys """
    result = {}
    for key in keys:
        if key in d: result[key] = d[key]
    return result

def get_constant(variable):
    """ Little hack to return the python value of a theano shared variable """
    return theano.function([], variable, mode=theano.compile.Mode(linker='py'))()

def get_value(variable):
    # TODO: I'm not sure this is the best way to do it
    if isinstance(variable, tensor.sharedvar.TensorSharedVariable):
        return variable.value
    else:
        return variable

def sharedX(value, name=None, borrow=False):
    """Transform value into a shared variable of type floatX"""
    return theano.shared(theano._asarray(value, dtype=floatX),
            name=name, borrow=False)

def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


##################################################
# Datasets and contest facilities
##################################################

def load_data(conf):
    """
    Loads a specified dataset according to the parameters in the dictionary
    """
    expected = ['normalize',
                'normalize_on_the_fly',
                'randomize_valid',
                'randomize_test']
    train_set, valid_set, test_set = load_ndarray_dataset(conf['dataset'], **subdict(conf, expected))

    # Allocate shared variables
    def shared_dataset(data_x):
        """Function that loads the dataset into shared variables"""
        if conf.get('normalize', True):
            return sharedX(data_x, borrow=True)
        else:
            return theano.shared(theano._asarray(data_x), borrow=True)

    if conf.get('normalize_on_the_fly', False):
        return [train_set, valid_set, test_set]
    else:
        test_set_x = shared_dataset(test_set)
        valid_set_x = shared_dataset(valid_set)
        train_set_x = shared_dataset(train_set)
        return [train_set_x, valid_set_x, test_set_x]


def create_submission(conf, get_representation):
    """
    Create submission files given a configuration dictionary and a computation function
    """
    # Load the dataset, without permuting valid and test
    kwargs = subdict(conf, ['dataset', 'normalize', 'normalize_on_the_fly'])
    kwargs.update(randomize_valid=False, randomize_test=False)
    train_set, valid_set, test_set = load_data(kwargs)
    
    # Valid and test representations
    valid_repr = get_representation(get_value(valid_set))
    test_repr = get_representation(get_value(test_set))

    # If there are too much features, outputs kernel matrices
    if (valid_repr.shape[1] > valid_repr.shape[0]):
        valid_repr = numpy.dot(valid_repr, valid_repr.T)
        test_repr = numpy.dot(test_repr, test_repr.T)

    # Quantitize data
    valid_repr = numpy.floor((valid_repr / valid_repr.max())*999)
    test_repr = numpy.floor((test_repr / test_repr.max())*999)

    # Convert into text info
    valid_text = ''
    test_text = ''

    for i in xrange(valid_repr.shape[0]):
        for j in xrange(valid_repr.shape[1]):
            valid_text += '%s ' % int(valid_repr[i, j])
        valid_text += '\n'
    del valid_repr

    for i in xrange(test_repr.shape[0]):
        for j in xrange(test_repr.shape[1]):
            test_text += '%s ' % int(test_repr[i, j])
        test_text += '\n'
    del test_repr

    # Write it in a .txt file
    basename = os.path.join(conf['submission_dir'],
                            conf['dataset'] + '_' + conf['expname'])
    valid_file = open(basename + '_valid.prepro', 'w')
    test_file = open(basename + '_final.prepro', 'w')

    valid_file.write(valid_text)
    test_file.write(test_text)

    valid_file.close()
    test_file.close()

    print >> sys.stderr, "... done creating files"

    os.system('zip %s %s %s' % (basename + '.zip',
                                basename + '_valid.prepro',
                                basename + '_final.prepro'))

    print >> sys.stderr, "... files compressed"

    os.system('rm %s %s' % (basename + '_valid.prepro',
                            basename + '_final.prepro'))

    print >> sys.stderr, "... useless files deleted"


##################################################
# Iterator objet for blending datasets
##################################################

class BatchIterator(object):
    """
    Builds an iterator object that can be used to go through the minibatches
    of a dataset, with respect to the given proportions in conf
    """
    def __init__(self, conf, dataset):
        # Compute maximum number of examples for training.
        set_proba = [conf['train_prop'], conf['valid_prop'], conf['test_prop']]
        set_sizes = [get_constant(data.shape[0]) for data in dataset]
        total_size = numpy.dot(set_proba, set_sizes)

        # Record other parameters.
        self.batch_size = conf['batch_size']
        self.dataset = [get_value(data) for data in dataset]

        # Upper bounds for minibatch indexes
        self.limit = [size // self.batch_size for size in set_sizes]
        self.counter = [0, 0, 0]
        self.index = 0
        self.max_index = total_size // self.batch_size

        # Sampled random number generator
        pairs = izip(set_sizes, set_proba)
        sampling_proba = [s * p / float(total_size) for s, p in pairs]
        cumulative_proba = numpy.cumsum(sampling_proba)
        def _generator():
            x = numpy.random.random()
            return numpy.searchsorted(cumulative_proba, x)

        self.generator = _generator
        # TODO: Record seed parameter for reproductability purposes ?

    def __iter__(self):
        """ Return a fresh self objet to iterate over all minibatches """
        self.counter = [0, 0, 0]
        self.index = 0
        return self

    def next(self):
        """
        Return the next minibatch for training according to sampling probabilities
        """
        if (self.index > self.max_index):
            raise StopIteration
        else:
            # Retrieve minibatch from chosen set
            chosen = self.generator()
            offset = self.counter[chosen]
            minibatch = self.dataset[chosen][offset * self.batch_size:(offset + 1) * self.batch_size]
            # Increment counters
            self.index += 1
            self.counter[chosen] = (self.counter[chosen] + 1) % self.limit[chosen]
            # Return the computed minibatch
            return minibatch
