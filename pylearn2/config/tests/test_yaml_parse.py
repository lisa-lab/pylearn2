import cPickle
import os
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

def test_import():
    loaded = load("a: !import 'decimal.Decimal'")
    assert_(loaded['a'] == Decimal)

def test_import_string():
    loaded = load("a: !import decimal.Decimal")
    assert_(loaded['a'] == Decimal)

def test_import_colon():
    loaded = load("a: !import:decimal.Decimal")
    assert_(loaded['a'] == Decimal)

def test_env_rhs():
    environ['TEST_VAR'] = '10'
    loaded = load("a: ${TEST_VAR}")
    assert_(loaded['a'] == 10)
    del environ['TEST_VAR']

def test_env_lhs():
    environ['TEST_VAR'] = 'a'
    loaded = load("${TEST_VAR}: 42")
    assert_(loaded['a'] == 42)
    del environ['TEST_VAR']

def test_env_pkl():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        d = ('a', 1)
        cPickle.dump(d, f)
    environ['TEST_VAR'] = fname
    loaded = load("a: !pkl: ${TEST_VAR}")
    assert_(loaded['a'] == d)
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

