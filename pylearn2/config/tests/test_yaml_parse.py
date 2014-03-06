import os
import numpy as np
import cPickle
import tempfile
from numpy.testing import assert_
from pylearn2.config.yaml_parse import load, load_path
from os import environ
from decimal import Decimal

def test_load_path():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write("a: 23")
    loaded = load_path(fname)
    assert_(loaded['a'] == 23)
    os.remove(fname)

def test_obj():
    loaded = load("a: !obj:decimal.Decimal { value : '1.23' }")
    assert_(isinstance(loaded['a'], Decimal))

def test_floats():
    loaded = load("a: { a: -1.23, b: 1.23e-1 }")
    assert_(isinstance(loaded['a']['a'], float))
    assert_(isinstance(loaded['a']['b'], float))
    assert_( (loaded['a']['a'] +1.23) < 1e-3 )
    assert_( (loaded['a']['b'] -1.23e-1) < 1e-3)

def test_import():
    loaded = load("a: !import 'decimal.Decimal'")
    assert_(loaded['a'] == Decimal)

def test_import_string():
    loaded = load("a: !import decimal.Decimal")
    assert_(loaded['a'] == Decimal)

def test_import_colon():
    loaded = load("a: !import:decimal.Decimal")
    assert_(loaded['a'] == Decimal)

def test_preproc_rhs():
    environ['TEST_VAR'] = '10'
    loaded = load('a: "${TEST_VAR}"')
    print "loaded['a'] is %s" % loaded['a']
    assert_(loaded['a'] == "10")
    del environ['TEST_VAR']

def test_preproc_pkl():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        d = ('a', 1)
        cPickle.dump(d, f)
    environ['TEST_VAR'] = fname
    loaded = load('a: !pkl: "${TEST_VAR}"')
    assert_(loaded['a'] == d)
    del environ['TEST_VAR']

def test_late_preproc_pkl():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        array = np.arange(10)
        np.save(f, array)
    environ['TEST_VAR'] = fname
    loaded = load('a: !obj:pylearn2.datasets.npy_npz.NpyDataset { file: "${TEST_VAR}" }\n')
    assert_( loaded['a'].yaml_src.find("${TEST_VAR}") != -1 )   # Assert the unsubstituted TEST_VAR is in yaml_src
    del environ['TEST_VAR']

def test_unpickle():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        d = {'a': 1, 'b': 2}
        cPickle.dump(d, f)
    loaded = load("{'a': !pkl: '%s'}" % fname)
    assert_(loaded['a'] == d)
    os.remove(fname)

def test_unpickle_key():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        d = ('a', 1)
        cPickle.dump(d, f)
    loaded = load("{!pkl: '%s': 50}" % fname)
    assert_(loaded.keys()[0] == d)
    assert_(loaded.values()[0] == 50)
    os.remove(fname)

