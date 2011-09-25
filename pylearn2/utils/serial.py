import cPickle
import pickle
import numpy as np
import os
import time
import warnings
import sys
from pylearn2.utils.string import preprocess
from cPickle import BadPickleGet
io = None
hdf_reader = None

def load(filepath, recurse_depth = 0):
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
            if str(nei).find('Please use HDF reader for matlab v7.3 files') != -1:
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
            print 'Max number of tries exceeded while trying to open '+filepath
            print 'attempting to open via reading string'
            f = open(filepath,'rb')
            lines = f.readlines()
            f.close()
            content = ''.join(lines)
            return cPickle.loads(content)
        else:
            nsec = 0.5 * (2.0 ** float(recurse_depth))
            print "Waiting "+str(nsec)+" seconds and trying again"
            time.sleep(nsec)
            return load(filepath, recurse_depth + 1)

    try:
        f = open(filepath,'rb')
        obj = cPickle.load(f)
        f.close()
        return obj
    except BadPickleGet, e:
        print 'Failed to open '+filepath+' due to BadPickleGet with string '+str(e)

        return exponential_backoff()
    except EOFError, e:
        print "Failed to open '+filepath+' due to EOFError with string "+str(e)

        return exponential_backoff()
    except ValueError, e:
        print 'Failed to open '+filepath+' due to ValueError with string '+str(e)

        return exponential_backoff()
    except Exception, e:
        #assert False
        exc_str = str(e)
        if len(exc_str) > 0:
            import pdb
            tb = pdb.traceback.format_exc()

            raise Exception("Couldn't open '"+filepath+"' due to: "+str(type(e))+','+str(e)+". Orig traceback:\n"+tb)
        else:
            print "Couldn't open '"+filepath+"' and exception has no string. Opening it again outside the try/catch so you can see whatever error it prints on its own."
            f = open(filepath,'rb')
            obj = cPickle.load(f)
            f.close()
            return obj
        #
    #
#


def save(filepath, obj):

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

            The workaround is just to raise the max recursion depth.
            Obviously this does not scale and could cause a crash
            but I don't see another solution short of writing our
            own implementation of pickle.
        """
        if str(e).find('recursion') != -1:
            warnings.warn('pylearn2.utils.save encountered the following error: ' \
                    + str(e) + \
                    '\nAttempting to resolve this error by calling ' + \
                    'sys.setrecusionlimit and retrying')

            sys.setrecursionlimit(50000)
            _save(filepath, obj)


def _save(filepath,obj):
        try:
                f = open(filepath,"wb")
        except Exception, e:
                raise Exception('failed to open '+filepath+' for writing, error is '+str(e))
        ""
        try:
            cPickle.dump(obj,f)
            f.close()
        except Exception, e:
            f.close()

            if str(e).find('maximum recursion depth exceeded') != -1:
                raise

            try:
                f = open(filepath,"wb")
                pickle.dump(obj,f)
                f.close()
            except Exception, e2:
                try:
                    f.close()
                except:
                    pass
                if str(e) == '' and str(e2) == '':
                    print 'neither cPickle nor pickle could write to '+filepath
                    print 'moreover, neither of them raised an exception that can be converted to a string'
                    print 'now re-attempting to write with cPickle outside the try/catch loop so you can see if it prints anything when it dies'
                    f = open(filepath,'wb')
                    cPickle.dump(obj,f)
                    f.close()
                    print 'Somehow or other, the file write worked once we quit using the try/catch.'
                else:
                    if str(e2) == 'env':
                        raise

                    import pdb
                    tb = pdb.traceback.format_exc()

                    raise Exception(str(obj)+
                                    ' could not be written to '+filepath+
                                    ' by cPickle due to '+str(e)+
                                    ' nor by pickle due to '+str(e2)+
                                    '. \nTraceback '+tb)
            print 'Warning: '+filepath+' was written by pickle instead of cPickle, due to '+str(e)+' (perhaps your object is eally big?)'
#

def clone_via_serialize(obj):
	str = cPickle.dumps(obj)
	return cPickle.loads(str)

def to_string(obj):
    return cPickle.dumps(obj)

def parent_dir(filepath):
        return '/'.join(filepath.split('/')[:-1])

def mkdir(filepath):
	try:
		os.makedirs(filepath)
	except:
		print "couldn't make directory '"+filepath+"', maybe it already exists"
