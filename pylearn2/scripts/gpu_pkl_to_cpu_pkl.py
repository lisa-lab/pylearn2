#!/usr/bin/env python
"""
Converts a pickle file containing CudaNdarraySharedVariables into
a pickle file containing only TensorSharedVariables.

Usage:

gpu_pkl_to_cpu_pkl.py <gpu.pkl> <cpu.pkl>

Loads gpu.pkl, replaces cuda shared variables with numpy ones,
and saves to cpu.pkl.

If you create a model while using GPU and later want to unpickle it
on a machine without a GPU, you must convert it this way.

This is theano's fault, not pylearn2's. I would like to fix theano,
but don't understand the innards of theano well enough, and none of
the theano developers has been willing to help me at all with this
issue. If it annoys you that you have to do this, please help me
persuade the theano developers that this issue is worth more of their
attention.

Note: This script is also useful if you want to create a model on GPU,
save it, and then run other theano functionality on CPU later, even
if your machine has a GPU. It could be useful to modify this script
to do the reverse conversion, so you can create a model on CPU, save
it, and then run theano functions on GPU later.

Further note: this script is very hacky and imprecise. It is likely
to do things like blow away subclasses of list and dict and turn them
into plain lists and dicts. It is also liable to overlook all sorts of
theano shared variables if you have an exotic data structure stored in
the pickle. You probably want to test that the cpu pickle file can be
loaded on a machine without GPU to be sure that the script actually
found them all.
"""
from __future__ import print_function

__author__ = "Ian Goodfellow"

import sys
import types

if __name__ == '__main__':
    _, in_path, out_path = sys.argv
    from pylearn2.utils import serial
    from theano import shared
    model = serial.load(in_path)

# map ids of objects we've fixed before to the fixed version, so we don't clone objects when fixing
# can't use object itself as key because not all objects are hashable
    already_fixed = {}

# ids of objects being fixed right now (we don't support cycles)
    currently_fixing = []

    blacklist = ["im_class", "func_closure", "co_argcount", "co_cellvars", "func_code",
            "append", "capitalize", "im_self", "func_defaults", "func_name"]
    blacklisted_keys = ["bytearray", "IndexError", "isinstance", "copyright", "main"]

    postponed_fixes = []

    class Placeholder(object):
        def __init__(self, id_to_sub):
            self.id_to_sub = id_to_sub

    class FieldFixer(object):

        def __init__(self, obj, field, fixed_field):
            self.obj = obj
            self.field = field
            self.fixed_field = fixed_field

        def apply(self):
            obj = self.obj
            field = self.field
            fixed_field = already_fixed[self.fixed_field.id_to_sub]
            setattr(obj, field, fixed_field)

    def fix(obj, stacklevel=0):
        prefix = ''.join(['.']*stacklevel)
        oid = id(obj)
        canary_oid = oid
        print(prefix + 'fixing '+str(oid))
        if oid in already_fixed:
            return already_fixed[oid]
        if oid in currently_fixing:
            print('returning placeholder for '+str(oid))
            return Placeholder(oid)
        currently_fixing.append(oid)
        if hasattr(obj, 'set_value'):
            # Base case: we found a shared variable, must convert it
            rval = shared(obj.get_value())
            # Sabotage its getstate so if something tries to pickle it, we'll find out
            obj.__getstate__ = None
        elif obj is None:
            rval = None
        elif isinstance(obj, list):
            print(prefix + 'fixing a list')
            rval = []
            for i, elem in enumerate(obj):
                print(prefix + '.fixing elem %d' % i)
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
        elif isinstance(obj, dict):
            print(prefix + 'fixing a dict')
            rval = obj
            """
            rval = {}
            for key in obj:
                if key in blacklisted_keys or (isinstance(key, str) and key.endswith('Error')):
                    print(prefix + '.%s is blacklisted' % str(key))
                    rval[key] = obj[key]
                    continue
                print(prefix + '.fixing key ' + str(key) + ' of type '+str(type(key)))
                fixed_key = fix(key, stacklevel + 2)
                if isinstance(fixed_key, Placeholder):
                    raise NotImplementedError()
                print(prefix + '.fixing value for key '+str(key))
                fixed_value = fix(obj[key], stacklevel + 2)
                if isinstance(fixed_value, Placeholder):
                    raise NotImplementedError()
                rval[fixed_key] = fixed_value
            """
        elif isinstance(obj, tuple):
            print(prefix + 'fixing a tuple')
            rval = []
            for i, elem in enumerate(obj):
                print(prefix + '.fixing elem %d' % i)
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
            rval = tuple(rval)
        elif isinstance(obj, (int, float, str)):
            rval = obj
        else:
            print(prefix + 'fixing a generic object')
            field_names = dir(obj)
            for field in field_names:
                if isinstance(getattr(obj, field), types.MethodType):
                    print(prefix + '.%s is an instancemethod' % field)
                    continue
                if field in blacklist or (field.startswith('__')):
                    print(prefix + '.%s is blacklisted' % field)
                    continue
                print(prefix + '.fixing field %s' % field)
                updated_field = fix(getattr(obj, field), stacklevel + 2)
                print(prefix + '.applying fix to field %s' % field)
                if isinstance(updated_field, Placeholder):
                    postponed_fixes.append(FieldFixer(obj, field, updated_field))
                else:
                    try:
                        setattr(obj, field, updated_field)
                    except Exception as e:
                        print("Couldn't do that because of exception: "+str(e))
            rval = obj
        already_fixed[oid] = rval
        print(prefix+'stored fix for '+str(oid))
        assert canary_oid == oid
        del currently_fixing[currently_fixing.index(oid)]
        return rval

    model = fix(model)

    assert len(currently_fixing) == 0

    for fixer in postponed_fixes:
        fixer.apply()

    serial.save(out_path, model)

