import cPickle
import pickle
import numpy as np
import os
import time
import warnings
import sys
from pylearn2.utils.string_utils import preprocess
from cPickle import BadPickleGet
io = None
hdf_reader = None
import struct


def load(filepath, recurse_depth=0):

    try:
        import joblib
        joblib_available = True
    except ImportError:
        joblib_available = False
    if recurse_depth == 0:
        filepath = preprocess(filepath)

    if filepath.endswith('.npy'):
        return np.load(filepath)

    if filepath.endswith('.mat'):
        global io
        if io is None:
            import scipy.io
            io = scipy.io
        try:
            return io.loadmat(filepath)
        except NotImplementedError, nei:
            if str(nei).find('HDF reader') != -1:
                global hdf_reader
                if hdf_reader is None:
                    import h5py
                    hdf_reader = h5py
                return hdf_reader.File(filepath)
            else:
                raise
        #this code should never be reached
        assert False

    def exponential_backoff():
        if recurse_depth > 9:
            print ('Max number of tries exceeded while trying to open ' +
                   filepath)
            print 'attempting to open via reading string'
            f = open(filepath, 'rb')
            lines = f.readlines()
            f.close()
            content = ''.join(lines)
            return cPickle.loads(content)
        else:
            nsec = 0.5 * (2.0 ** float(recurse_depth))
            print "Waiting " + str(nsec) + " seconds and trying again"
            time.sleep(nsec)
            return load(filepath, recurse_depth + 1)

    try:
        if not joblib_available:
            with open(filepath, 'rb') as f:
                obj = cPickle.load(f)
        else:
            obj = joblib.load(filepath)

    except BadPickleGet, e:
        print ('Failed to open ' + str(filepath) +
               ' due to BadPickleGet with exception string ' + str(e))

        obj =  exponential_backoff()
    except EOFError, e:
        print ('Failed to open ' + str(filepath) +
               ' due to EOFError with exception string ' + str(e))

        obj =  exponential_backoff()
    except ValueError, e:
        print ('Failed to open ' + str(filepath) +
               ' due to ValueError with string ' + str(e))

        obj =  exponential_backoff()
    except Exception, e:
        #assert False
        exc_str = str(e)
        if len(exc_str) > 0:
            import pdb
            tb = pdb.traceback.format_exc()
            raise Exception("Couldn't open '" + str(filepath) +
                            "' due to: " + str(type(e)) + ', ' + str(e) +
                            ". Orig traceback:\n" + tb)
        else:
            print ("Couldn't open '" + str(filepath) +
                   "' and exception has no string. Opening it again outside "
                   "the try/catch so you can see whatever error it prints on "
                   "its own.")
            f = open(filepath, 'rb')
            obj = cPickle.load(f)
            f.close()

    #if the object has no yaml_src, we give it one that just says it
    #came from this file. could cause trouble if you save obj again
    #to a different location
    if not hasattr(obj,'yaml_src'):
        try:
            obj.yaml_src = '!pkl: "'+os.path.abspath(filepath)+'"'
        except:
            pass

    return obj


def save(filepath, obj):
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
    """
    filepath = preprocess(filepath)
    try:
        _save(filepath, obj)
    except RuntimeError, e:
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
            warnings.warn('pylearn2.utils.save encountered the following '
                          'error: ' + str(e) +
                          '\nAttempting to resolve this error by calling ' +
                          'sys.setrecusionlimit and retrying')
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(50000)
                _save(filepath, obj)
            finally:
                sys.setrecursionlimit(old_limit)


def _save(filepath, obj):
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
                cPickle.dump(obj, filehandle)
    except Exception, e:
        # TODO: logging, or warning
        print "cPickle has failed to write an object to " + filepath
        if str(e).find('maximum recursion depth exceeded') != -1:
            raise
        try:
            # TODO: logging, or warning
            print 'retrying with pickle'
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
        except Exception, e2:
            if str(e) == '' and str(e2) == '':
                # TODO: logging, or warning
                print (
                    'neither cPickle nor pickle could write to %s' % filepath
                )
                print (
                    'moreover, neither of them raised an exception that '
                    'can be converted to a string'
                )
                print (
                    'now re-attempting to write with cPickle outside the '
                    'try/catch loop so you can see if it prints anything '
                    'when it dies'
                )
                with open(filepath, 'wb') as f:
                    cPickle.dump(obj, f)
                print ('Somehow or other, the file write worked once '
                       'we quit using the try/catch.')
            else:
                if str(e2) == 'env':
                    raise

                import pdb
                tb = pdb.traceback.format_exc()
                raise IOError(str(obj) +
                              ' could not be written to '+
                              str(filepath) +
                              ' by cPickle due to ' + str(e) +
                              ' nor by pickle due to ' + str(e2) +
                              '. \nTraceback '+ tb)
        print ('Warning: ' + str(filepath) +
               ' was written by pickle instead of cPickle, due to '
               + str(e) +
               ' (perhaps your object is really big?)')


def clone_via_serialize(obj):
    str = cPickle.dumps(obj)
    return cPickle.loads(str)


def to_string(obj):
    return cPickle.dumps(obj)


def mkdir(filepath):
    try:
        os.makedirs(filepath)
    except:
        print ("couldn't make directory '" + str(filepath) +
               "', maybe it already exists")

def read_int( fin, n = 1):
    if n == 1:
        return struct.unpack('i', fin.read(4))[0]
    else:
        rval = []
        for i in xrange(n):
            rval.append(read_int(fin))
        return rval

#dictionary to convert lush binary matrix magic numbers
#to dtypes
lush_magic = {
            507333717 : 'uint8',
            507333716 : 'int32',
            507333713 : 'float32',
            507333715 : 'float64'
        }

def read_bin_lush_matrix(filepath):
    f = open(filepath,'rb')
    magic = read_int(f)
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
        raise ValueError('Unrecognized lush magic number '+str(magic))

    rval = np.fromfile(file = f, dtype = dtype, count = total_elems)

    rval = rval.reshape(*shape)

    f.close()

    return rval
