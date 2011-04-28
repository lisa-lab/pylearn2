import cPickle
import pickle
import os
import time
import numpy as N


def load(filepath, recurse_depth = 0):
    try:
        f = open(filepath,'rb')
        obj = cPickle.load(f)
        f.close()
        return obj
    except ValueError, e:
        print 'Failed to open '+filepath+' due to ValueError with string '+str(e)
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
        #
    except Exception, e:
        #assert False
        exc_str = str(e)
        if len(exc_str) > 0:
            raise Exception("Couldn't open '"+filepath+"' due to: "+str(type(e))+','+str(e))
        else:
            print "Couldn't open '"+filepath+"' and exception has no string. Opening it again outside the try/catch so you can see whatever error it prints on its own."
            f = open(filepath,'rb')
            obj = cPickle.load(f)
            f.close()
            return obj
        #
    #
#

def save(filepath,obj):
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
                    raise Exception(str(obj)+' could not be written to '+filepath+' by cPickle due to '+str(e)+' nor by pickle due to '+str(e2))
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
