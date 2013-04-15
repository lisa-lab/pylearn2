import cPickle
import os
import tempfile
from numpy.testing import assert_
from pylearn2.config.yaml_parse import load


def test_unpickle():
    fd, fname = tempfile.mkstemp()
    f = os.fdopen(fd, 'wb')
    d = {'a': 1, 'b': 2}
    cPickle.dump(d, f)
    f.close()
    loaded = load("{'a': !pkl: '%s'}" % fname)
    assert_(loaded['a'] == d)
    os.remove(fname)


def test_unpickle_key():
    fd, fname = tempfile.mkstemp()
    f = os.fdopen(fd, 'wb')
    d = ('a', 1)
    cPickle.dump(d, f)
    f.close()
    loaded = load("{!pkl: '%s': 50}" % fname)
    assert_(loaded.keys()[0] == d)
    assert_(loaded.values()[0] == 50)
    os.remove(fname)
