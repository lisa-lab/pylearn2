"""
Several utilities for experimenting upon utlc datasets
"""
# Standard library imports
import os
import inspect
import zipfile
from tempfile import TemporaryFile

# Third-party imports
import numpy
import theano
from pylearn.datasets.utlc import load_ndarray_dataset, load_sparse_dataset
from pylearn2.utils import subdict, sharedX, embed


##################################################
# Shortcuts and auxiliary functions
##################################################


"""
this function marked deprecated by Ian Goodfellow
TODO: find all uses of this function and remove it,
      OR change the comment to explain why it needs to exist
         if it is just intended to be used on shared variables,
         shouldn't we just use get_value?

def get_constant(variable, return_scalar=False):
     Little hack to return the python value of a theano shared variable.

    If return_scalar is True, and the constant is an ndarray of 0 dimensions,
    the content of that ndarray will be returned instead.

    try:
        ret = theano.function([],
                              variable,
                              mode=theano.compile.Mode(linker='py')
                              )()
    except TypeError:
        ret = variable
    if isinstance(ret, numpy.ndarray) and ret.ndim == 0:
        ret = ret.item()
    return ret
"""



def getboth(dict1, dict2, key, default=None):
    """
    Try to retrieve key from dict1 if exists, otherwise try with dict2.
    If the key is not found in any of them, raise an exception.
    """
    try:
        return dict1[key]
    except KeyError:
        if default is None:
            return dict2[key]
        else:
            return dict2.get(key, default)

##################################################
# Datasets loading and contest facilities
##################################################

def load_data(conf):
    """
    Loads a specified dataset according to the parameters in the dictionary
    """
    print '... loading dataset'

    # Special case for sparse format
    if conf.get('sparse', False):
        expected = inspect.getargspec(load_sparse_dataset)[0][1:]
        data = load_sparse_dataset(conf['dataset'], **subdict(conf, expected))
        valid, test = data[1:3]

        # Sparse TERRY data on LISA servers contains an extra null first row in
        # valid and test subsets.
        if conf['dataset'] == 'terry':
            valid = valid[1:]
            test = test[1:]
            assert valid.shape[0] == test.shape[0] == 4096, \
                'Sparse TERRY data loaded has wrong number of examples'

        if len(data) == 3:
            return [data[0], valid, test]
        else:
            return [data[0], valid, test, data[3]]

    # Load as the usual ndarray
    expected = inspect.getargspec(load_ndarray_dataset)[0][1:]
    data = load_ndarray_dataset(conf['dataset'], **subdict(conf, expected))

    # Special case for on-the-fly normalization
    if conf.get('normalize_on_the_fly', False):
        return data

    # Allocate shared variables
    def shared_dataset(data_x):
        """Function that loads the dataset into shared variables"""
        if conf.get('normalize', True):
            return sharedX(data_x, borrow=True)
        else:
            return theano.shared(theano._asarray(data_x), borrow=True)

    return map(shared_dataset, data)


def save_submission(conf, valid_repr, test_repr):
    """
    Create a submission file given a configuration dictionary and a
    representation for valid and test.
    """
    print '... creating zipfile'

    # Ensure the given directory is correct
    submit_dir = conf['savedir']
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    elif not os.path.isdir(submit_dir):
        raise IOError('savedir %s is not a directory' % submit_dir)

    basename = os.path.join(submit_dir, conf['dataset'] + '_' + conf['expname'])

    # If there are too much features, outputs kernel matrices
    if (valid_repr.shape[1] > valid_repr.shape[0]):
        valid_repr = numpy.dot(valid_repr, valid_repr.T)
        test_repr = numpy.dot(test_repr, test_repr.T)

    # Quantitize data
    valid_repr = numpy.floor((valid_repr / valid_repr.max())*999)
    test_repr = numpy.floor((test_repr / test_repr.max())*999)

    # Store the representations in two temporary files
    valid_file = TemporaryFile()
    test_file = TemporaryFile()

    numpy.savetxt(valid_file, valid_repr, fmt="%.3f")
    numpy.savetxt(test_file, test_repr, fmt="%.3f")

    # Reread those files and put them together in a .zip
    valid_file.seek(0)
    test_file.seek(0)

    submission = zipfile.ZipFile(basename + ".zip", "w",
                                 compression=zipfile.ZIP_DEFLATED)
    submission.writestr(basename + '_valid.prepro', valid_file.read())
    submission.writestr(basename + '_final.prepro', test_file.read())

    submission.close()
    valid_file.close()
    test_file.close()

def create_submission(conf, transform_valid, transform_test=None, features=None):
    """
    Create a submission file given a configuration dictionary and a
    computation function.

    Note that it always reload the datasets to ensure valid & test
    are not permuted.
    """
    if transform_test is None:
        transform_test = transform_valid

    # Load the dataset, without permuting valid and test
    kwargs = subdict(conf, ['dataset', 'normalize', 'normalize_on_the_fly', 'sparse'])
    kwargs.update(randomize_valid=False, randomize_test=False)
    valid_set, test_set = load_data(kwargs)[1:3]

    # Sparse datasets are not stored as Theano shared vars.
    if not conf.get('sparse', False):
        valid_set = valid_set.get_value(borrow=True)
        test_set = test_set.get_value(borrow=True)

    # Prefilter features, if needed.
    if features is not None:
        valid_set = valid_set[:, features]
        test_set = test_set[:, features]

    # Valid and test representations
    valid_repr = transform_valid(valid_set)
    test_repr = transform_test(test_set)

    # Convert into text info
    save_submission(conf, valid_repr, test_repr)

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

    # Concatenate the sets, and give different one hot labels for valid and test
    n_valid = valid_repr.shape[0]
    n_test = test_repr.shape[0]

    _labvalid = numpy.hstack((numpy.ones((n_valid, 1)),
                              numpy.zeros((n_valid, 1))))
    _labtest = numpy.hstack((numpy.zeros((n_test, 1)),
                             numpy.ones((n_test, 1))))

    dataset = numpy.vstack((valid_repr, test_repr))
    label = numpy.vstack((_labvalid, _labtest))

    print '... computing the ALC'
    return embed.score(dataset, label)


def lookup_alc(data, transform):
    valid_repr = transform(data[1].get_value(borrow=True))
    test_repr = transform(data[2].get_value(borrow=True))

    return compute_alc(valid_repr, test_repr)
