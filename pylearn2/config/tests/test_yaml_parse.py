"""
Unit tests for ./yaml_parse.py
"""

import os
import numpy as np
import cPickle
import tempfile
from numpy.testing import assert_
from pylearn2.config.yaml_parse import load, load_path, initialize
from os import environ, close
from decimal import Decimal
from tempfile import mkstemp
from pylearn2.utils import serial
from pylearn2.utils.exc import reraise_as
import yaml


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
    assert_((loaded['a']['a'] + 1.23) < 1e-3)
    assert_((loaded['a']['b'] - 1.23e-1) < 1e-3)


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
    loaded = load('a: !obj:pylearn2.datasets.npy_npz.NpyDataset '
                  '{ file: "${TEST_VAR}"}\n')
    # Assert the unsubstituted TEST_VAR is in yaml_src
    assert_(loaded['a'].yaml_src.find("${TEST_VAR}") != -1)
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


def test_multi_constructor_obj():
    """
    Tests whether multi_constructor_obj throws an exception when
    the keys in mapping are None.
    """
    try:
        load("a: !obj:decimal.Decimal { 1 }")
    except TypeError as e:
        assert str(e) == "Received non string object (1) as key in mapping."
        pass
    except Exception, e:
        error_msg = "Got the unexpected error: %s" % (e)
        reraise_as(ValueError(error_msg))


def test_duplicate_keywords():
    """
    Tests whether there are doublicate keywords in the yaml
    """
    initialize()
    yamlfile = """{
            "model": !obj:pylearn2.models.mlp.MLP {
            "layers": [
                     !obj:pylearn2.models.mlp.Sigmoid {
                         "layer_name": 'h0',
                         "dim": 20,
                         "sparse_init": 15,
                     }],
            "nvis": 784,
            "nvis": 384,
        }
    }"""

    try:
        load(yamlfile)
    except yaml.constructor.ConstructorError, e:
        message = str(e)
        assert message.endswith("found duplicate key (nvis)")
        pass
    except Exception, e:
        error_msg = "Got the unexpected error: %s" % (e)
        raise TypeError(error_msg)


def test_duplicate_keywords_2():
    """
    Tests whether duplicate keywords as independent parameters works fine.
    """
    initialize()
    yamlfile = """{
             "model": !obj:pylearn2.models.rbm.GaussianBinaryRBM {

                 "vis_space" : &vis_space !obj:pylearn2.space.Conv2DSpace {
                    "shape" : [32,32],
                    "num_channels" : 3
                },
                "hid_space" : &hid_space !obj:pylearn2.space.Conv2DSpace {
                    "shape" : [27,27],
                    "num_channels" : 10
                },
                "transformer" :
                        !obj:pylearn2.linear.conv2d.make_random_conv2D {
                    "irange" : .05,
                    "input_space" : *vis_space,
                    "output_space" : *hid_space,
                    "kernel_shape" : [6,6],
                    "batch_size" : &batch_size 5
                },
                "energy_function_class" :
                     !obj:pylearn2.energy_functions.rbm_energy.grbm_type_1 {},
                "learn_sigma" : True,
                "init_sigma" : .3333,
                "init_bias_hid" : -2.,
                "mean_vis" : False,
                "sigma_lr_scale" : 1e-3

             }
    }"""

    load(yamlfile)


def test_parse_null_as_none():
    """
    Tests whether None may be passed via yaml kwarg null.
    """
    initialize()
    yamlfile = """{
             "model": !obj:pylearn2.models.autoencoder.Autoencoder {

                 "nvis" : 1024,
                 "nhid" : 64,
                 "act_enc" : Null,
                 "act_dec" : null

             }
    }"""
    load(yamlfile)


class DumDum(object):
    pass


def test_pkl_yaml_src_field():
    """
    Tests a regression where yaml_src wasn't getting correctly set on pkls.
    """
    try:
        fd, fn = mkstemp()
        close(fd)
        o = DumDum()
        o.x = ('a', 'b', 'c')
        serial.save(fn, o)
        yaml = '!pkl: \'' + fn + '\'\n'
        loaded = load(yaml)
        assert loaded.x == ('a', 'b', 'c')
        assert loaded.yaml_src == yaml
    finally:
        os.remove(fn)


def test_instantiate_regression():
    """
    Test a case where instantiated objects were not getting correctly
    reused across references.
    """
    yaml = ("{'a': &test !obj:pylearn2.config.tests.test_yaml_parse.DumDum {},"
            " 'b': *test}")
    obj = load(yaml)
    assert obj['a'] is obj['b']


if __name__ == "__main__":
    test_multi_constructor_obj()
    test_duplicate_keywords()
    test_duplicate_keywords_2()
    test_unpickle_key()
