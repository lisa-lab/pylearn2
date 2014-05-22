#!/usr/bin/env python
"""
Determines the contribution of different subcomponents of a file to its file size, serialization time,
and deserialization time.
"""
import sys
from pylearn2.utils import serial
import cPickle
import pickle
import time
from theano.printing import min_informative_str

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

def usage():
    """
    .. todo::

        WRITEME
    """
    print """
Usage:
first argument is a cPickle file to load
if no more arguments are supplied, will analyze each field of the root-level object stored in the file
subsequent arguments let you index into fields / dictionary entries of the object
For example,
pkl_inspector.py foo.pkl .my_field [my_key] 3
will load an object obj from foo.pkl and analyze obj.my_field["my_key"][3]
"""

if __name__ == "__main__":
    if len(sys.argv) == 1:
        usage()
        sys.exit(-1)

    hp = pickle.HIGHEST_PROTOCOL

    filepath = sys.argv[1]

    orig_obj = serial.load(filepath)

    cycle_check = {}

    obj_name = 'root_obj'
    cycle_check[id(orig_obj)] = obj_name

    for field in sys.argv[2:]:
        if field.startswith('['):
            assert field.endswith(']')
            obj_name += '[' + field[1:-1] + ']'
            orig_obj = orig_obj[field[1:-1]]
        elif field.startswith('.'):
            obj_name += '.' + field
            orig_obj = getattr(orig_obj,field[1:])
        else:
            obj_name + '[' + field + ']'
            orig_obj = orig_obj[eval(field)]
        if id(orig_obj) in cycle_check:
            print "You're going in circles, "+obj_name+" is the same as "+cycle_check[id(orig_obj)]
            quit()
        cycle_check[id(orig_obj)] = obj_name


    print 'type of object: '+str(type(orig_obj))
    print 'object: '+str(orig_obj)
    print 'object, longer description:\n'+min_informative_str(orig_obj, indent_level = 1)

    t1 = time.time()
    s = cPickle.dumps(orig_obj, hp)
    t2 = time.time()
    prev_ts = t2 - t1

    prev_bytes = len(s)
    print 'orig_obj bytes: \t\t\t\t'+str(prev_bytes)
    t1 = time.time()
    x = cPickle.loads(s)
    t2 = time.time()
    prev_t = t2 - t1
    print 'orig load time: '+str(prev_t)
    print 'orig save time: '+str(prev_ts)

    idx = 0

    print 'field\t\t\tdelta bytes\t\t\tdelta load time\t\t\tdelta save time'

    if not (isinstance(orig_obj, dict) or isinstance(orig_obj, list) ):
        while len(dir(orig_obj)) > idx:
            stop = False

            while True:
                fields = dir(orig_obj)
                if idx >= len(fields):
                    stop = True
                    break
                field = fields[idx]

                if field in ['names_to_del','__dict__']:
                    print 'not deleting '+field
                    idx += 1
                    continue

                success = True
                try:
                    delattr(orig_obj,field)

                except:
                    #TODO: add a config flag to allow printing the following messages
                    #print "got error trying to delete "+field
                    idx += 1
                    success = False
                if success and field in dir(orig_obj):
                    print field + ' reappears after being deleted'
                    idx += 1
                if success:
                    break

            if stop:
                break

            #print hasattr(orig_obj, 'names_to_del')
            t1 = time.time()
            s = cPickle.dumps(orig_obj, hp)
            t2 = time.time()
            new_ts = t2 - t1
            diff_ts = prev_ts - new_ts
            prev_ts = new_ts
            new_bytes = len(s)
            diff_bytes = prev_bytes - new_bytes
            prev_bytes = new_bytes
            t1 = time.time()
            x = cPickle.loads(s)
            t2 = time.time()
            new_t = t2 - t1
            diff_t = prev_t - new_t
            prev_t = new_t
            print field+': \t\t\t\t'+str(diff_bytes)+'\t\t\t'+str(diff_t)+'\t\t\t'+str(diff_ts)

    if isinstance(orig_obj, dict):
        print 'orig_obj is a dictionary'

        keys = [ key for key in orig_obj.keys() ]

        for key in keys:
            orig_obj[key] = None

            s = cPickle.dumps(orig_obj, hp)
            new_bytes = len(s)
            t1 = time.time()
            x = cPickle.loads(s)
            t2 = time.time()
            new_t = t2 - t1
            diff_t = prev_t - new_t
            prev_t = new_t
            diff_bytes = prev_bytes - new_bytes
            prev_bytes = new_bytes
            print 'val for '+str(key)+': \t\t\t\t'+str(diff_bytes)+'\t\t\t'+str(diff_t)

        for key in keys:
            del orig_obj[key]

            s = cPickle.dumps(orig_obj, hp)
            new_bytes = len(s)
            t1 = time.time()
            x = cPickle.loads(s)
            t2 = time.time()
            new_t = t2 - t1
            diff_t = prev_t - new_t
            prev_t = new_t
            diff_bytes = prev_bytes - new_bytes
            prev_bytes = new_bytes
            print str(key)+': \t\t\t\t'+str(diff_bytes)+'\t\t\t'+str(diff_t)


    if isinstance(orig_obj, list):
        print 'orig_obj is a list'

        i = 0
        while len(orig_obj) > 0:
            stringrep = str(orig_obj[0])
            if len(stringrep) > 15:
                stringrep = stringrep[0:12] + "..."
            del orig_obj[0]

            t1 = time.time()
            s = cPickle.dumps(orig_obj, hp)
            t2 = time.time()
            new_ts = t2 - t1
            diff_ts = prev_ts - new_ts
            prev_ts = new_ts

            new_bytes = len(s)
            diff_bytes = prev_bytes - new_bytes
            prev_bytes = new_bytes

            t1 = time.time()
            x = cPickle.loads(s)
            t2 = time.time()
            new_t = t2 - t1
            diff_t = prev_t - new_t
            prev_t = new_t
            print stringrep+': \t\t\t\t'+str(diff_bytes)+'\t\t\t'+str(diff_t)+'\t\t\t'+str(diff_ts)

            i+= 1

