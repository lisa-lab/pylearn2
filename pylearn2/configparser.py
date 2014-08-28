__author__ = 'denadai2'

import os
import pylearn2

from theano.configparser import (TheanoConfigParser, THEANO_FLAGS_DICT, theano_cfg, theano_raw_cfg, ConfigParser,
                                 TypedParam, AddConfigVar, TheanoConfigWarning,
                                _config_var_list, config)

pylearn2_config = TheanoConfigParser()

def fetch_val_for_key2(key):
    """Return the overriding config value for a key.
    A successful search returns a string value.
    An unsuccessful search raises a KeyError
    The (decreasing) priority order is:
    - PYLEARN2_FLAGS
    - THEANO_FLAGS
    - ~./theanorc
    """
    # first try to find it in the FLAGS

    try:
        return PYLEARN2_FLAGS_DICT[key]
    except KeyError:
        pass
    # next try to find it in the config file
    # config file keys can be of form option, or section.option
    key_tokens = key.split('.')
    if len(key_tokens) > 2:
        raise KeyError(key)
    if len(key_tokens) == 2:
        section, option = key_tokens
    else:
        section, option = 'global', key
    try:
        try:
            return theano_cfg.get(section, option)
        except ConfigParser.InterpolationError:
            return theano_raw_cfg.get(section, option)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError(key)


""" NOTE: parse_config_string is here only because of a bug in theano. In this moment in Theano 0.6 there is
    for kv_pair in THEANO_FLAGS.split(','):
    and not
    for kv_pair in config_string.split(','):
    It should be removed with the corrected version
"""
def parse_config_string(config_string, issue_warnings=True):
    """
    Parses a config string (comma-separated key=value components) into a dict.
    """
    config_dict = {}
    for kv_pair in config_string.split(','):
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split('=', 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                TheanoConfigWarning.warn(
                    ("Config key '%s' has no value, ignoring it"
                     % kv_tuple[0]),
                    stacklevel=1)
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict

class ConfigParam(object):

    def __init__(self, default, filter=None, allow_override=True):
        """
        If allow_override is False, we can't change the value after the import
        of Theano. So the value should be the same during all the execution.
        """
        self.default = default
        self.filter = filter
        self.allow_override = allow_override
        # N.B. --
        # self.fullname  # set by AddConfigVar
        # self.doc       # set by AddConfigVar

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in AddConfigVar, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.

    def __get__(self, *args):
        if not hasattr(self, 'val'):
            try:
                val_str = fetch_val_for_key2(self.fullname)
            except KeyError:
                if callable(self.default):
                    val_str = self.default()
                else:
                    val_str = self.default
            self.__set__(None, val_str)
        #print "RVAL", self.val
        return self.val

    def __set__(self, cls, val):
        if not self.allow_override and hasattr(self, 'val'):
            raise Exception(
                    "Can't change the value of this config parameter "
                    "after initialization!")
        #print "SETTING PARAM", self.fullname,(cls), val
        if self.filter:
            self.val = self.filter(val)
        else:
            self.val = val

class TypedParam(ConfigParam):
    def __init__(self, default, mytype, is_valid=None, allow_override=True):
        self.mytype = mytype

        def filter(val):
            cast_val = mytype(val)
            if callable(is_valid):
                if is_valid(cast_val):
                    return cast_val
                else:
                    raise ValueError(
                            'Invalid value (%s) for configuration variable '
                            '"%s".'
                            % (val, self.fullname), val)
            return cast_val

        super(TypedParam, self).__init__(default, filter,
                allow_override=allow_override)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.mytype)

def AddPylearnConfigVar(name, doc, configparam, root=pylearn2_config, in_c_key=True):
    """
    This is a wrapper of theano.configparser.AddConfigVar. See the official theano documentation.
    """
    # This method also performs some of the work of initializing ConfigParam
    # instances
    if root is pylearn2_config:
        # only set the name in the first call, not the recursive ones
        configparam.fullname = name

    AddConfigVar(name, doc, configparam, root, in_c_key)


def SemicolonParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, str, is_valid, allow_override=allow_override)

PYLEARN2_FLAGS = os.getenv("PYLEARN2_FLAGS", "")
PYLEARN2_FLAGS_DICT = parse_config_string(PYLEARN2_FLAGS, issue_warnings=True)