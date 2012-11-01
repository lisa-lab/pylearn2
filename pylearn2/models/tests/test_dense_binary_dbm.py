import pylearn2
from pylearn2.models.dense_binary_dbm import load_matlab_dbm
from pylearn2.models.dense_binary_dbm import InferenceProcedure
import warnings
try:
    from scipy import io
except ImportError:
    warnings.warn("Could not import scipy")
import numpy as np
from theano import config
import theano.tensor as T
from theano import function
from pylearn2.testing.skip import skip_if_no_scipy

def test_dbm_loader():
    """ Loads an example model and some data and makes
    sure that inference gets the same result as Ruslan
    Salakhutdino's demo code. """
    skip_if_no_scipy()
    pylearn2_path = pylearn2.__path__[0]
    dbm_path = pylearn2_path + '/models/tests/dbm.mat'
    data_path = pylearn2_path + '/models/tests/dbm_data.mat'

    dbm = load_matlab_dbm(dbm_path)

    d = io.loadmat( data_path )

    for key in d:
        try:
            d[key] = np.cast[config.floatX](d[key])
        except:
            pass

    ip = InferenceProcedure( layer_schedule = [0,1] * 10 )
    ip.register_model(dbm)

    V = T.matrix()
    if config.compute_test_value != 'off':
        V.tag.test_value = d['data']
    h1_theano, h2_theano = ip.infer(V)['H_hat']

    f = function([V],[h1_theano,h2_theano])

    data = d['data']

    h1_numpy, h2_numpy = f(data)

    assert np.allclose(d['h1'],h1_numpy)
    assert np.allclose(d['h2'],h2_numpy)

if __name__ == "__main__":
    test_dbm_loader()
