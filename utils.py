"""
Several utilities for experimenting upon utlc datasets
"""
# Standard library imports
import os
from itertools import repeat

# Third-party imports
import numpy
import theano
from pylearn.datasets.utlc import load_ndarray_dataset

# Local imports
from auc import embed

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

def sharedX(value, name=None, borrow=False):
    """Transform value into a shared variable of type floatX"""
    return theano.shared(theano._asarray(value, dtype=floatX),
                         name=name,
                         borrow=False)

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
    print '... loading data'
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
    
    Note that it always reload the datasets to ensure valid & test are not permuted
    """
    # Load the dataset, without permuting valid and test
    kwargs = subdict(conf, ['dataset', 'normalize', 'normalize_on_the_fly'])
    kwargs.update(randomize_valid=False, randomize_test=False)
    train_set, valid_set, test_set = load_data(kwargs)
    
    # Valid and test representations
    valid_repr = get_representation(get_constant(valid_set))
    test_repr = get_representation(test_set.get_value())

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
    submit_dir = conf['submit_dir']
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    elif not os.path.isdir(submit_dir):
        raise IOError('submit_dir %s is not a directory' % submit_dir)

    basename = os.path.join(submit_dir, conf['dataset'] + '_' + conf['expname'])
    valid_file = open(basename + '_valid.prepro', 'w')
    test_file = open(basename + '_final.prepro', 'w')

    valid_file.write(valid_text)
    test_file.write(test_text)

    valid_file.close()
    test_file.close()

    print "... done creating files"

    os.system('zip -j %s %s %s' % (basename + '.zip',
                                   basename + '_valid.prepro',
                                   basename + '_final.prepro'))

    print "... files compressed"

    os.system('rm %s %s' % (basename + '_valid.prepro',
                            basename + '_final.prepro'))

    print "... useless files deleted"

##################################################
# Proxies for representation evaluations
##################################################

def compute_alc(valid_repr, test_repr):
    """
    Returns the ALC of the valid set VS test set
    Note: This proxy won't work in the case of transductive learning
    (This is an assumption) but it seems to be a good proxy in the
    normal case (i.e only train on training set)
    """

    # Concatenate the two sets, and give different one hot labels for valid and test
    n_valid  = valid_repr.shape[0]
    n_test = test_repr.shape[0]

    _labvalid = numpy.hstack((numpy.ones((n_valid,1)), numpy.zeros((n_valid,1))))
    _labtest = numpy.hstack((numpy.zeros((n_test,1)), numpy.ones((n_test,1))))

    dataset = numpy.vstack((valid_repr, test_repr))
    label = numpy.vstack((_labvalid,_labtest))
    
    print '... computing the ALC'
    return embed.score(dataset, label)

def lookup_alc(data, transform):
    valid_repr = transform(data[1].get_value())
    test_repr = transform(data[2].get_value())
    
    return compute_alc(valid_repr, test_repr)

##################################################
# Iterator objet for blending datasets
##################################################

class BatchIterator(object):
    """
    Builds an iterator object that can be used to go through the minibatches
    of a dataset, with respect to the given proportions in conf
    """
    def __init__(self, conf, dataset):
        # Record external parameters
        self.batch_size = conf['batch_size']
        # TODO: If you have a better way to return dataset slices, I'll take it
        self.dataset = [set.get_value() for set in dataset]
        
        # Compute maximum number of samples for one loop
        set_proba = [conf['train_prop'], conf['valid_prop'], conf['test_prop']]
        set_sizes = [get_constant(data.shape[0]) for data in dataset]
        set_batch = [float(self.batch_size) for i in xrange(3)]
        set_range = numpy.divide(numpy.multiply(set_proba, set_sizes), set_batch)
        set_range = map(int, numpy.ceil(set_range))

        # Upper bounds for each minibatch indexes
        set_limit = numpy.ceil(numpy.divide(set_sizes, set_batch))
        self.limit = map(int, set_limit)
        
        # Number of rows in the resulting union
        flo = numpy.floor
        sub = numpy.subtract
        mul = numpy.multiply
        div = numpy.divide
        mod = numpy.mod
        l_trun = mul(flo(div(set_range, set_limit)), mod(set_sizes, set_batch))
        l_full = mul(sub(set_range, flo(div(set_range, set_limit))), set_batch)
        
        self.length = sum(l_full) + sum(l_trun)

        # Random number generation using a permutation
        index_tab = []
        for i in xrange(3):
            index_tab.extend(repeat(i, set_range[i]))

        # The seed should be deterministic
        self.seed = conf.get('batchiterator_seed', 300)
        rng = numpy.random.RandomState(seed=self.seed)
        self.permut = rng.permutation(index_tab)

    def __iter__(self):
        """ Generator function to iterate through all minibatches """
        counter = [0, 0, 0]
        for chosen in self.permut:
            # Retrieve minibatch from chosen set
            index = counter[chosen]
            minibatch = self.dataset[chosen][index * self.batch_size:(index + 1) * self.batch_size]
            # Increment the related counter
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            # Return the computed minibatch
            yield minibatch

    def __len__(self):
        """ Return length of the weighted union """
        return self.length

    def by_index(self):
        """ Same generator as __iter__, but yield only the chosen indexes """
        counter = [0, 0, 0]
        for chosen in self.permut:
            index = counter[chosen]
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            yield chosen, index

##################################################
# Miscellaneous
##################################################

def blend(conf, data):
    """ Randomized blending of datasets in data according to parameters in conf """
    iterator = BatchIterator(conf, data)
    nrow = len(iterator)
    ncol = data[0].get_value().shape[1]
    array = numpy.empty((nrow, ncol), data[0].dtype)
    index = 0
    for minibatch in iterator:
        for row in minibatch:
            array[index] = row
            index += 1
            
    return sharedX(array, borrow=True)
