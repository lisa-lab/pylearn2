#!/usr/bin/env python
# coding: utf-8

"""
prediction code for classification, without using batches
(it's simpler that way)
if you run out of memory it could be resolved by implementing a batch version
the model should be an MLP
see http://fastml.com/how-to-get-predictions-from-pylearn2/
author: Zygmunt Zając
"""

import sys
import os
import numpy as np
import cPickle as pickle

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
        test_path = sys.argv[2]
        out_path = sys.argv[3]
    except IndexError:
        print "Usage: predict.py <model file> <test file> <output file>"
        print "       predict.py saved_model.pkl test_x.csv predictions.csv\n"
        quit()

    print "loading model..."

    try:
        model = serial.load(model_path)
    except Exception, e:
        print "error loading {}:".format(model_path)
        print e
        quit(-1)

    print "setting up symbolic expressions..."

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

# drop this when performing regression
    Y = T.argmax(Y, axis=1)

    f = function([X], Y)

    print "loading data and predicting..."

# x is a numpy array
# x = pickle.load(open(test_path, 'rb'))
    x = np.loadtxt(test_path, delimiter=',')	# no labels in the file

    y = f(x)

    print "writing predictions..."

    np.savetxt(out_path, y, fmt='%d')


