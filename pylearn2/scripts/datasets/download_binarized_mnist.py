"""
Download script for the unlabeled version of the MNIST dataset, used in

    On the Quantitative Analysis of Deep Belief Networks
    Salakhutdinov and Murray
    http://www.mit.edu/~rsalakhu/papers/dbn_ais.pdf
    The MNIST database of handwritten digits
    LeCun and Cortes
    http://yann.lecun.com/exdb/mnist/
"""
from __future__ import print_function

__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import os
import urllib
import numpy

assert 'PYLEARN2_DATA_PATH' in os.environ, "PYLEARN2_DATA_PATH not defined"
mnist_path = os.path.join(os.environ['PYLEARN2_DATA_PATH'], "binarized_mnist")

if not os.path.isdir(mnist_path):
    print("creating path: " + mnist_path)
    os.makedirs(mnist_path)

in_dir = os.listdir(mnist_path)
mnist_files = ["binarized_mnist_train", "binarized_mnist_valid",
               "binarized_mnist_test"]
base_url = "http://www.cs.toronto.edu/~larocheh/public/datasets/" + \
           "binarized_mnist/"

if not all([f + ".npy" in in_dir for f in mnist_files]) or in_dir == []:
    print("Downloading MNIST data...")
    npy_out = [os.path.join(mnist_path, f + ".npy") for f in mnist_files]
    mnist_url = ["".join([base_url, f, ".amat"]) for f in mnist_files]

    for n_out, m_url in zip(npy_out, mnist_url):
        print("Downloading " + m_url + "...", end='')
        numpy.save(n_out, numpy.loadtxt(urllib.urlretrieve(m_url)[0]))
        print(" Done")

    print("Done downloading MNIST")
else:
    print("MNIST files already in PYLEARN2_DATA_PATH")
