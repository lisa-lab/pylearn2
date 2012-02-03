#!/bin/env python
#argument: path to a pkl file
#loads the pkl file and figures out which fields are CudaNDArrays

import sys

path = sys.argv[1]

from pylearn2.utils import serial

obj = serial.load(path)

from theano.sandbox.cuda import CudaNdarray

def find(cur_obj, cur_name):
    if isinstance(cur_obj, CudaNdarray):
        print cur_name
    for field in dir(cur_obj):
        find(getattr(cur_obj,field),cur_name+'.'+field)

find(obj,'')
