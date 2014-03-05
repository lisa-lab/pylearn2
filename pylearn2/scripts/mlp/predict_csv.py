#!/usr/bin/env python
# coding: utf-8

"""
prediction code for classification, without using batches
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

try:
	model_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
except IndexError:
	print "Usage: predict.py <model file> <test file> <output file>"
	print "       predict.py saved_clf.pkl test_x.csv predictions.csv\n"
	quit()
	
print "loading model..."

try:
	model = serial.load(model_path)
except Exception, e:
	print model_path + "doesn't seem to be a valid model path, got this error when trying to load it:"
	print e

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


