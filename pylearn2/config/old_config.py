import yaml
import inspect
import types

from pylearn2.utils.exc import reraise_as

global resolvers

def load(json_file_path):
    """ given a file path to a json file, returns the dictionary definde by the json file """

    f = open(json_file_path)
    lines = f.readlines()
    f.close()

    content = ''.join(lines)

    return yaml.load(content)

def get_field(d, key):
    try:
        rval = d[key]
    except KeyError:
        reraise_as(ValueError('Could not access "'+key+'" in \n'+str(d)))
    return rval

def get_str(d, key):
    rval = get_field(d, key)

    if not isinstance(rval,str):
        raise TypeError('"'+key+'" entry is not a string in the following: \n'+str(d))

    return rval

def get_tag(d):
    return get_str(d, 'tag')

def resolve(d):
    """ given a dictionary d, returns the object described by the dictionary """

    tag = get_tag(d)

    try:
        resolver = resolvers[tag]
    except KeyError:
        reraise_as(TypeError('config does not know of any object type "'+tag+'"'))

    return resolver(d)

def resolve_model(d):
    assert False

def resolve_dataset(d):
    import pylearn2.datasets.config
    return pylearn2.datasets.config.resolve(d)

def resolve_train_algorithm(d):
    assert False

resolvers = {
        'model'             : resolve_model,
        'dataset'           : resolve_dataset,
        'train_algorithm'   : resolve_train_algorithm
        }


