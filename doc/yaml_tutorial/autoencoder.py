from __future__ import print_function

import numpy
import pickle

class AutoEncoder:

    def __init__(self, nvis, nhid, iscale=0.1,
            activation_fn=numpy.tanh,
            params=None):

        self.nvis = nvis
        self.nhid = nhid
        self.activation_fn = activation_fn

        if params is None:
            self.W = iscale * numpy.random.randn(nvis, nhid)
            self.bias_vis = numpy.zeros(nvis)
            self.bias_hid = numpy.zeros(nhid)
        else:
            self.W = params[0]
            self.bias_vis = params[1]
            self.bias_hid = params[2]

    def __str__(self):
        rval  = '%s\n' % self.__class__.__name__
        rval += '\tnvis = %i\n' % self.nvis
        rval += '\tnhid = %i\n' % self.nhid
        rval += '\tactivation_fn = %s\n' % str(self.activation_fn)
        rval += '\tmean std(weights) = %.2f\n' % self.W.std(axis=0).mean()
        return rval

    def save(self, fname):
        fp = open(fname, 'w')
        pickle.dump([self.W, self.bias_vis, self.bias_hid], fp)
        fp.close()


if __name__ == '__main__':
    import os
    from StringIO import StringIO
    from pylearn2.config import yaml_parse

    example1 = """
        !obj:yaml_tutorial.autoencoder.AutoEncoder {
           "nvis": 784,
           "nhid": 100,
           "iscale": 0.2,
        }
    """
    stream = StringIO()
    stream.write(example1)
    stream.seek(0)
    print('Example 1: building basic auto-encoder.')
    model = yaml_parse.load(stream)
    print(model)
    stream.close()

    example2 = """
        !obj:yaml_tutorial.autoencoder.AutoEncoder {
           "nvis": &nvis 100,
           "nhid": *nvis,
        }
    """
    stream = StringIO()
    stream.write(example2)
    stream.seek(0)
    print('Example 2: anchors and references.')
    model = yaml_parse.load(stream)
    print(model)
    stream.close()

    example3 = """
        !obj:yaml_tutorial.autoencoder.AutoEncoder {
           "nvis": 784,
           "nhid": 100,
           "iscale": 1.0,
           "activation_fn": !import 'pylearn2.expr.nnet.sigmoid_numpy', 
        }
    """
    stream = StringIO()
    stream.write(example3)
    stream.seek(0)
    print('Example 3: dynamic imports through !import.')
    model = yaml_parse.load(stream)
    model.save('example3_weights.pkl')
    print(model)
    stream.close()

    example4 = """
        !obj:yaml_tutorial.autoencoder.AutoEncoder {
           "nvis": 784,
           "nhid": 100,
           "params": !pkl: 'example3_weights.pkl',
        }
    """
    stream = StringIO()
    stream.write(example4)
    stream.seek(0)
    print('Example 4: loading data with !pkl command.')
    model = yaml_parse.load(stream)
    print(model)
    stream.close()

