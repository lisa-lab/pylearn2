"""
Utilities for serializing and deserializing python objects.
"""
try:
    from cPickle import BadPickleGet
except ImportError:
    BadPickleGet = KeyError
import pickle
import logging
import numpy as np
from theano.compat import six
from theano.compat.six.moves import cPickle, xrange
import os
import time
import warnings
import sys
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.mem import improve_memory_error_message
io = None
hdf_reader = None
import struct
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.string_utils import match
import shutil

logger = logging.getLogger(__name__)


def load(filepath, retry=True):
    """
    Loads object(s) from file specified by 'filepath'.

    Parameters
    ----------
    filepath : str
        A path to a file to load. Should be a pickle, Matlab, or NumPy
        file; or a .txt or .amat file that numpy.loadtxt can load.
    retry : bool, optional
        If True, will make a handful of attempts to load the file before
        giving up. This can be useful if you are for example calling
        show_weights.py on a file that is actively being written to by a
        training script--sometimes the load attempt might fail if the
        training script writes at the same time show_weights tries to
        read, but if you try again after a few seconds you should be able
        to open the file.

    Returns
    -------
    loaded_object : object
        The object that was stored in the file.
    """

    return _load(filepath, recurse_depth=0, retry=True)


def save(filepath, obj, on_overwrite='ignore'):
    """
    Serialize `object` to a file denoted by `filepath`.

    Parameters
    ----------
    filepath : str
        A filename. If the suffix is `.joblib` and joblib can be
        imported, `joblib.dump` is used in place of the regular
        pickling mechanisms; this results in much faster saves by
        saving arrays as separate .npy files on disk. If the file
        suffix is `.npy` than `numpy.save` is attempted on `obj`.
        Otherwise, (c)pickle is used.

    obj : object
        A Python object to be serialized.

    on_overwrite : str, optional
        A string specifying what to do if the file already exists.
        Possible values include:

        - "ignore" : Just overwrite the existing file.
        - "backup" : Make a backup copy of the file (<filepath>.bak).
          Save the new copy. Then delete the backup copy. This allows
          recovery of the old version of the file if saving the new one
          fails.
    """
    filepath = preprocess(filepath)

    if os.path.exists(filepath):
        if on_overwrite == 'backup':
            backup = filepath + '.bak'
            shutil.move(filepath, backup)
            save(filepath, obj)
            try:
                os.remove(backup)
            except Exception as e:
                warnings.warn("Got an error while trying to remove " + backup
                              + ":" + str(e))
            return
        else:
            assert on_overwrite == 'ignore'

    try:
        _save(filepath, obj)
    except RuntimeError as e:
        """ Sometimes for large theano graphs, pickle/cPickle exceed the
            maximum recursion depth. This seems to me like a fundamental
            design flaw in pickle/cPickle. The workaround I employ here
            is the one recommended to someone who had a similar problem
            on stackexchange:

            http://stackoverflow.com/questions/2134706/hitting-maximum-recursion-depth-using-pythons-pickle-cpickle

            Obviously this does not scale and could cause a crash
            but I don't see another solution short of writing our
            own implementation of pickle.
        """
        if str(e).find('recursion') != -1:
            logger.warning('pylearn2.utils.save encountered the following '
                           'error: ' + str(e) +
                           '\nAttempting to resolve this error by calling ' +
                           'sys.setrecusionlimit and retrying')
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(50000)
                _save(filepath, obj)
            finally:
                sys.setrecursionlimit(old_limit)


def get_pickle_protocol():
    """
    Allow configuration of the pickle protocol on a per-machine basis.

    This way, if you use multiple platforms with different versions of
    pickle, you can configure each of them to use the highest protocol
    supported by all of the machines that you want to be able to
    communicate.
    """
    try:
        protocol_str = os.environ['PYLEARN2_PICKLE_PROTOCOL']
    except KeyError:
        # If not defined, we default to 0 because this is the default
        # protocol used by cPickle.dump (and because it results in
        # maximum portability)
        protocol_str = '0'
    if protocol_str == 'pickle.HIGHEST_PROTOCOL':
        return pickle.HIGHEST_PROTOCOL
    return int(protocol_str)


def _save(filepath, obj):
    """
    .. todo::

        WRITEME
    """
    try:
        import joblib
        joblib_available = True
    except ImportError:
        joblib_available = False
    if filepath.endswith('.npy'):
        np.save(filepath, obj)
        return
    # This is dumb
    # assert filepath.endswith('.pkl')
    save_dir = os.path.dirname(filepath)
    # Handle current working directory case.
    if save_dir == '':
        save_dir = '.'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise IOError("save path %s exists, not a directory" % save_dir)
    elif not os.access(save_dir, os.W_OK):
        raise IOError("permission error creating %s" % filepath)
    try:
        if joblib_available and filepath.endswith('.joblib'):
            joblib.dump(obj, filepath)
        else:
            if filepath.endswith('.joblib'):
                warnings.warn('Warning: .joblib suffix specified but joblib '
                              'unavailable. Using ordinary pickle.')
            with open(filepath, 'wb') as filehandle:
                cPickle.dump(obj, filehandle, get_pickle_protocol())
    except Exception as e:
        logger.exception("cPickle has failed to write an object to "
                         "{0}".format(filepath))
        if str(e).find('maximum recursion depth exceeded') != -1:
            raise
        try:
            logger.info('retrying with pickle')
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e2:
            if str(e) == '' and str(e2) == '':
                logger.exception('neither cPickle nor pickle could write to '
                                 '{0}'.format(filepath))
                logger.exception(
                    'moreover, neither of them raised an exception that '
                    'can be converted to a string'
                )
                logger.exception(
                    'now re-attempting to write with cPickle outside the '
                    'try/catch loop so you can see if it prints anything '
                    'when it dies'
                )
                with open(filepath, 'wb') as f:
                    cPickle.dump(obj, f, get_pickle_protocol())
                logger.info('Somehow or other, the file write worked once '
                            'we quit using the try/catch.')
            else:
                if str(e2) == 'env':
                    raise

                import pdb
                tb = pdb.traceback.format_exc()
                reraise_as(IOError(str(obj) +
                                   ' could not be written to ' +
                                   str(filepath) +
                           ' by cPickle due to ' + str(e) +
                                   ' nor by pickle due to ' + str(e2) +
                                   '. \nTraceback ' + tb))
        logger.warning('{0} was written by pickle instead of cPickle, due to '
                       '{1} (perhaps your object'
                       ' is really big?)'.format(filepath, e))


def clone_via_serialize(obj):
    """
    Makes a "deep copy" of an object by serializing it and then
    deserializing it.

    Parameters
    ----------
    obj : object
        The object to clone.

    Returns
    -------
    obj2 : object
        A copy of the object.
    """
    s = cPickle.dumps(obj, get_pickle_protocol())
    return cPickle.loads(s)


def to_string(obj):
    """
    Serializes an object to a string.

    Parameters
    ----------
    obj : object
        The object to serialize.

    Returns
    -------
    string : str
        The object serialized as a string.
    """
    return cPickle.dumps(obj, get_pickle_protocol())


def from_string(s):
    """
    Deserializes an object from a string.

    Parameters
    ----------
    s : str
        The object serialized as a string.

    Returns
    -------
    obj : object
        The object.
    """
    return cPickle.loads(s)


def mkdir(filepath):
    """
    Make a directory.

    Should succeed even if it needs to make more than one
    directory and nest subdirectories to do so. Raises an error if the
    directory can't be made. Does not raise an error if the directory
    already exists.

    Parameters
    ----------
    filepath : WRITEME
    """
    try:
        os.makedirs(filepath)
    except OSError:
        if not os.path.isdir(filepath):
            raise


def read_int(fin, n=1):
    """
    Reads n ints from a file.

    Parameters
    ----------
    fin : file
        Readable file object
    n : int
        Number of ints to read

    Returns
    -------
    rval : int or list
        The integer or integers requested
    """
    if n == 1:
        s = fin.read(4)
        if len(s) != 4:
            raise ValueError('fin did not contain 4 bytes')
        return struct.unpack('i', s)[0]
    else:
        rval = []
        for i in xrange(n):
            rval.append(read_int(fin))
        return rval

# dictionary to convert lush binary matrix magic numbers
# to dtypes
lush_magic = {
    507333717: 'uint8',
    507333716: 'int32',
    507333713: 'float32',
    507333715: 'float64'
}


def read_bin_lush_matrix(filepath):
    """
    Reads a binary matrix saved by the lush library.

    Parameters
    ----------
    filepath : str
        The path to the file.

    Returns
    -------
    matrix : ndarray
        A NumPy version of the stored matrix.
    """
    f = open(filepath, 'rb')
    try:
        magic = read_int(f)
    except ValueError:
        reraise_as("Couldn't read magic number")
    ndim = read_int(f)

    if ndim == 0:
        shape = ()
    else:
        shape = read_int(f, max(3, ndim))

    total_elems = 1
    for dim in shape:
        total_elems *= dim

    try:
        dtype = lush_magic[magic]
    except KeyError:
        reraise_as(ValueError('Unrecognized lush magic number ' + str(magic)))

    rval = np.fromfile(file=f, dtype=dtype, count=total_elems)

    excess = f.read(-1)

    if excess:
        raise ValueError(str(len(excess)) +
                         ' extra bytes found at end of file.'
                         ' This indicates  mismatch between header '
                         'and content')

    rval = rval.reshape(*shape)

    f.close()

    return rval


def load_train_file(config_file_path, environ=None):
    """
    Loads and parses a yaml file for a Train object.
    Publishes the relevant training environment variables

    Parameters
    ----------
    config_file_path : str
        Path to a config file containing a YAML string describing a
        pylearn2.train.Train object
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables when parsing the YAML file. If a key appears
        both in `os.environ` and this dictionary, the value in this
        dictionary is used.


    Returns
    -------
    Object described by the YAML string stored in the config file
    """
    from pylearn2.config import yaml_parse

    suffix_to_strip = '.yaml'

    # Publish environment variables related to file name
    if config_file_path.endswith(suffix_to_strip):
        config_file_full_stem = config_file_path[0:-len(suffix_to_strip)]
    else:
        config_file_full_stem = config_file_path

    os.environ["PYLEARN2_TRAIN_FILE_FULL_STEM"] = config_file_full_stem

    directory = config_file_path.split('/')[:-1]
    directory = '/'.join(directory)
    if directory != '':
        directory += '/'
    os.environ["PYLEARN2_TRAIN_DIR"] = directory
    os.environ["PYLEARN2_TRAIN_BASE_NAME"] = config_file_path.split('/')[-1]
    os.environ[
        "PYLEARN2_TRAIN_FILE_STEM"] = config_file_full_stem.split('/')[-1]

    return yaml_parse.load_path(config_file_path, environ=environ)


def _load(filepath, recurse_depth=0, retry=True):
    """
    Recursively tries to load a file until success or maximum number of
    attempts.

    Parameters
    ----------
    filepath : str
        A path to a file to load. Should be a pickle, Matlab, or NumPy
        file; or a .txt or .amat file that numpy.loadtxt can load.
    recurse_depth : int, optional
        End users should not use this argument. It is used by the function
        itself to implement the `retry` option recursively.
    retry : bool, optional
        If True, will make a handful of attempts to load the file before
        giving up. This can be useful if you are for example calling
        show_weights.py on a file that is actively being written to by a
        training script--sometimes the load attempt might fail if the
        training script writes at the same time show_weights tries to
        read, but if you try again after a few seconds you should be able
        to open the file.

    Returns
    -------
    loaded_object : object
        The object that was stored in the file.
    """
    try:
        import joblib
        joblib_available = True
    except ImportError:
        joblib_available = False
    if recurse_depth == 0:
        filepath = preprocess(filepath)

    if filepath.endswith('.npy') or filepath.endswith('.npz'):
        return np.load(filepath)

    if filepath.endswith('.amat') or filepath.endswith('txt'):
        try:
            return np.loadtxt(filepath)
        except Exception:
            reraise_as("{0} cannot be loaded by serial.load (trying "
                       "to use np.loadtxt)".format(filepath))

    if filepath.endswith('.mat'):
        global io
        if io is None:
            import scipy.io
            io = scipy.io
        try:
            return io.loadmat(filepath)
        except NotImplementedError as nei:
            if str(nei).find('HDF reader') != -1:
                global hdf_reader
                if hdf_reader is None:
                    import h5py
                    hdf_reader = h5py
                return hdf_reader.File(filepath, 'r')
            else:
                raise
        # this code should never be reached
        assert False

    # for loading PY2 pickle in PY3
    encoding = {'encoding': 'latin-1'} if six.PY3 else {}

    def exponential_backoff():
        if recurse_depth > 9:
            logger.info('Max number of tries exceeded while trying to open '
                        '{0}'.format(filepath))
            logger.info('attempting to open via reading string')
            with open(filepath, 'rb') as f:
                content = f.read()
            return cPickle.loads(content, **encoding)
        else:
            nsec = 0.5 * (2.0 ** float(recurse_depth))
            logger.info("Waiting {0} seconds and trying again".format(nsec))
            time.sleep(nsec)
            return _load(filepath, recurse_depth + 1, retry)

    try:
        if not joblib_available:
            with open(filepath, 'rb') as f:
                obj = cPickle.load(f, **encoding)
        else:
            try:
                obj = joblib.load(filepath)
            except Exception as e:
                if os.path.exists(filepath) and not os.path.isdir(filepath):
                    raise
                raise_cannot_open(filepath)
    except MemoryError as e:
        # We want to explicitly catch this exception because for MemoryError
        # __str__ returns the empty string, so some of our default printouts
        # below don't make a lot of sense.
        # Also, a lot of users assume any exception is a bug in the library,
        # so we can cut down on mail to pylearn-users by adding a message
        # that makes it clear this exception is caused by their machine not
        # meeting requirements.
        if os.path.splitext(filepath)[1] == ".pkl":
            improve_memory_error_message(e,
                                         ("You do not have enough memory to "
                                          "open %s \n"
                                          " + Try using numpy.{save,load} "
                                          "(file with extension '.npy') "
                                          "to save your file. It uses less "
                                          "memory when reading and "
                                          "writing files than pickled files.")
                                         % filepath)
        else:
            improve_memory_error_message(e,
                                         "You do not have enough memory to "
                                         "open %s" % filepath)

    except (BadPickleGet, EOFError, KeyError) as e:
        if not retry:
            reraise_as(e.__class__('Failed to open {0}'.format(filepath)))
        obj = exponential_backoff()
    except ValueError:
        logger.exception

        if not retry:
            reraise_as(ValueError('Failed to open {0}'.format(filepath)))
        obj = exponential_backoff()
    except Exception:
        # assert False
        reraise_as("Couldn't open {0}".format(filepath))

    # if the object has no yaml_src, we give it one that just says it
    # came from this file. could cause trouble if you save obj again
    # to a different location
    if not hasattr(obj, 'yaml_src'):
        try:
            obj.yaml_src = '!pkl: "' + os.path.abspath(filepath) + '"'
        except Exception:
            pass

    return obj


def raise_cannot_open(path):
    """
    Raise an exception saying we can't open `path`.

    Parameters
    ----------
    path : str
        The path we cannot open
    """
    pieces = path.split('/')
    for i in xrange(1, len(pieces) + 1):
        so_far = '/'.join(pieces[0:i])
        if not os.path.exists(so_far):
            if i == 1:
                if so_far == '':
                    continue
                reraise_as(IOError('Cannot open ' + path + ' (' + so_far +
                           ' does not exist)'))
            parent = '/'.join(pieces[0:i - 1])
            bad = pieces[i - 1]

            if not os.path.isdir(parent):
                reraise_as(IOError("Cannot open " + path + " because " +
                           parent + " is not a directory."))

            candidates = os.listdir(parent)

            if len(candidates) == 0:
                reraise_as(IOError("Cannot open " + path + " because " +
                           parent + " is empty."))

            if len(candidates) > 100:
                # Don't attempt to guess the right name if the directory is
                # huge
                reraise_as(IOError("Cannot open " + path + " but can open " +
                                   parent + "."))

            if os.path.islink(path):
                reraise_as(IOError(path + " appears to be a symlink to a "
                                   "non-existent file"))
            reraise_as(IOError("Cannot open " + path + " but can open " +
                       parent + ". Did you mean " + match(bad, candidates) +
                       " instead of " + bad + "?"))
        # end if
    # end for
    assert False
