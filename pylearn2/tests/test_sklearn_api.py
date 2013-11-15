"""
Tests for the sklearn_api
"""
__authors__ = "Alexandre Lacoste"
__copyright__ = "Copyright 2013, Universite Laval"
__credits__ = ["Alexandre Lacoste"]
__license__ = "3-clause BSD"
__maintainer__ = "Alexandre Lacoste"


from pylearn2.sklearn_api import make_dataset, MaxoutClassifier
import numpy as np

def _dummy_ds():
    x = np.array( [[0,1],[1,0],[1,0] ] )
    y = np.array( [-1, 1, 1 ] )
    return x,y

def test_make_dataset():
    x, y = _dummy_ds()
    _ds, classmap = make_dataset(x, y)
    assert set(classmap._map) == set([0,1])
    assert set(classmap._invmap) == set([-1,1])

def test_buid_maxout():
    maxout = MaxoutClassifier()
    ds, _classmap = make_dataset(* _dummy_ds())
    maxout._build_model(ds)

## this test is long and may randomly fail in some occasions
#def test_maxout_predict():
#    x,y =  _dummy_ds()
#    maxout = MaxoutClassifier(num_units = (4,)) # a really small architecture should be enough for this task
#    maxout.fit( x,y )
#    y_ = maxout.predict(x)
#    print y, y_
#    np.testing.assert_equal(y,y_)
    
