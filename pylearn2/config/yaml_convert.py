"""Code for converting our YAML experiment descriptions to JSON format."""
import sys
import re
import yaml
import json
import collections
import StringIO

from pylearn2.config.ordered_yaml_loader import OrderedDictYAMLLoader
from pylearn2.utils.call_check import checked_call
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.string_utils import match
import warnings


def multi_constructor_obj(loader, tag_suffix, node):
    """
    Converts "key: !obj:python.path { }," to "key: { __obj__: python.path }".
    """
    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)
    mapping['__obj__'] = tag_suffix
    return mapping


def multi_constructor_pkl(loader, tag_suffix, node):
    """
    Converts "key: !pkl: loadme.pkl" to "key: { __pkl__: loadme.pkl }".
    """
    mapping = loader.construct_yaml_str(node)
    return {"__pkl__": mapping}


def multi_constructor_import(loader, tag_suffix, node):
    """
    Converts "key: !import: python.path" to "key: { __import__: python.path }".
    """
    mapping = loader.construct_yaml_str(node)
    return {"__import__": mapping}


def key_value_events(key, value):
    """
    Generates a list of events which correspond to a key-value pair.
    """
    key_event = yaml.ScalarEvent(anchor=None, tag=None,
            implicit=(False, True), value=key)
    val_event = yaml.ScalarEvent(anchor=None, tag=None,
            implicit=(False, True), value=value)
    return [key_event, val_event]


def anchor_events(anchor, value):
    """
    Generates a list of events, which preserve the semantics of anchored scalar
    yaml variables, but with JSON's syntax. To do this, we translate "key:
    &anchor val" to "key: {'__anchor__': 'anchor', '__value__': value}".  In
    term of yaml events, this corresponds to modifying the sequence:

    ScalarEvent(anchor=None, ..., value=u'key')
    ScalarEvent(anchor=u'anchor', ..., value=3)

    to the following:

    ScalarEvent(anchor=None,  value=u'key')
    MappingStartEvent(anchor=None, ...)
    ScalarEvent(anchor=None, ..., value='__anchor__')
    ScalarEvent(anchor=None, ..., value=u'anchor')
    ScalarEvent(anchor=None, ..., value='__value__')
    ScalarEvent(anchor=None, ..., value=3)
    MappingEndEvent()
    """
    events = [yaml.MappingStartEvent(anchor=None, tag=None, implicit=True)]
    events += key_value_events('__anchor__', anchor)
    events += key_value_events('__value__', value)
    events += [yaml.MappingEndEvent()]
    return events


def alias_events(key, anchor):
    """
    Generates a list of events, which preserves the semantics of yaml
    references, but with the syntax of JSON. To do this, we translate "key:
    *anchor" to "key: {'__ref__': 'anchor'}". In term of yaml events, this
    corresponds to modifying the sequence:

        ScalarEvent(anchor=None, ..., value=u'key')
        AliasEvent(anchor=u'anchor')

    to the following:

        ScalarEvent(anchor=None, ..., value=u'key')
        MappingStartEvent(anchor=None, ..., implicit=True)
        ScalarEvent(anchor=None, ..., value='__ref__')
        ScalarEvent(anchor=None, ..., value=u'anchor')
        MappingEndEvent()
    """
    events = [yaml.MappingStartEvent(anchor=None, tag=None, implicit=True)]
    events += key_value_events(key, anchor)
    events += [yaml.MappingEndEvent()]
    return events


def strip_anchors(istream, ostream):
    """
    This method parses a yaml input stream, and outputs a similar stream where
    YAML anchors and references have been "flattened", such that the syntax is
    compatible with the JSON format. Concretely, we translate:

        key1: !tag:desc &anchor_name { attribute1: val1 },
        key2: *anchor_name,

    to:

        key1: !tag:desc &anchor {
            __anchor__: anchor_name,
            attribute1: val1,
        },
        key2: { __ref__: anchor_name},
    """
    events = []
    for event in yaml.parse(istream):
        if (type(event) is yaml.MappingStartEvent and event.anchor):
            anchor_event = key_value_events('__anchor__', event.anchor)
            event.anchor = None
            events += [event] + anchor_event
        elif (type(event) is yaml.ScalarEvent and event.anchor is not None):
            events += anchor_events(event.anchor, event.value)
        elif (type(event) is yaml.AliasEvent):
            events += alias_events('__ref__', event.anchor)
        else:
            events += [event]
    yaml.emit(events, stream=ostream)
    ostream.seek(0)


def yaml_to_json(istream):
    """
    Converts a Pylearn2 yaml config file into JSON format. It does this by
    converting special tags !tag:value to (__tag__,value) entries of the
    respective dictionaries. Anchors and references are also converted to
    (key,value) pairs, with keys taking the values "__anchor__" and "__ref__".

    See also: strip_anchors, alias_events, obj_constructor, pkl_constructor and
    import_constructor.
    """
    # make yaml string JSON compatible
    ostream = StringIO.StringIO()
    strip_anchors(istream, ostream)
    # parse to make sure we didn't break anything
    loader = OrderedDictYAMLLoader(ostream)
    loader.add_multi_constructor('!obj:', multi_constructor_obj)
    loader.add_multi_constructor('!pkl:', multi_constructor_pkl)
    loader.add_constructor('!import:', multi_constructor_import)
    loader.add_multi_constructor('!import', multi_constructor_import)

    try:
        rval = loader.get_single_data()
    finally:
        loader.dispose()

    import pdb; pdb.set_trace()
    return json.dumps(rval, sort_keys=False, indent=4)


def pprint(dict_, indent=0):
    """
    Utility function. Pretty prints a dictionary dict_.
    """
    if dict_ is None:
        return ''

    tab1 = ''.join(['    '] * indent)
    tab2 = ''.join(['    '] * (indent + 1))
    rval = '%s{\n' % tab1
    for k, v in dict_.iteritems():
        rval += '%s%s: %s,\n' % (tab2, k, v)
    rval += '%s}' % tab1
    return rval


def json_to_yaml(istream):
    """
    Performs the inverse operation to yaml_to_json.
    See yaml_to_json for details.
    """
    dict_ = json.load(istream, object_pairs_hook=collections.OrderedDict)
    import pdb; pdb.set_trace()

    def depth_first_search(dict_, indent=0):
        for k, v in dict_.iteritems():
            if isinstance(v, dict):
                dict_[k] = depth_first_search(v, indent + 1)

        if '__ref__' in dict_:
            ref_value = dict_.pop('__ref__')
            return str('*%s' % ref_value)
        elif '__import__' in dict_:
            return '!import %s' % str(dict_.pop('__import__'))
        elif '__pkl__' in dict_:
            return '!pkl: %s' % str(dict_.pop('__pkl__'))

        obj_str = ''
        anchor_str = ''
        if '__obj__' in dict_:
            obj_str = '!obj:%s ' % dict_.pop('__obj__')
        if '__anchor__' in dict_:
            if '__value__' in dict_:
                # anchor was on a scalarEvent
                anchor_str = '&%s %s' %\
                        (dict_.pop('__anchor__'),
                         dict_.pop('__value__'))
                dict_ = None
            else:
                # anchor was on a mapping event (object)
                anchor_str = '&%s ' % dict_.pop('__anchor__')
        rval = '%s%s\n%s' % (obj_str, anchor_str, pprint(dict_, indent))
        return rval

    return depth_first_search(dict_)


if __name__ == "__main__":
    # Demonstration of how to specify objects, reference them
    # later in the configuration, etc.
    fp = open(sys.argv[1])
    json_string = yaml_to_json(fp)
    fp.close()
    fp = open('test.json', 'w')
    fp.write(json_string)
    fp.close()
    fp = open('test.json', 'r')
    yaml_str = json_to_yaml(fp)
    fp.close()
    fp = open('test.yaml', 'w')
    fp.write(yaml_str)
    fp.close()
