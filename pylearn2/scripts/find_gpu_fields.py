#!/usr/bin/env python
"""
.. todo::

    WRITEME
"""
#argument: path to a pkl file
#loads the pkl file and figures out which fields are CudaNDArrays

from __future__ import print_function

import sys

if __name__ == "__main__":
    path = sys.argv[1]

    from pylearn2.utils import serial
    import inspect

    obj = serial.load(path)

    from theano.sandbox.cuda import CudaNdarray

    visited = set([])

    def find(cur_obj, cur_name):
        global visited

        if isinstance(cur_obj, CudaNdarray):
            print(cur_name)
        print(cur_name)
        for field, new_obj in inspect.getmembers(cur_obj):

            if new_obj in visited:
                continue

            visited = visited.union([new_obj])

            print(visited)

            find(new_obj,cur_name+'.'+field)

    find(obj,'')
