import yaml
import inspect
import types

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
    except:
        raise ValueError('Could not access "'+key+'" in \n'+str(d))
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
    except:
        raise TypeError('config does not know of any object type "'+tag+'"')

    return resolver(d)


def check(thing_to_call, d):
    """ raises an exception with as helpful as possible of an error message if d does not contain the right
    keywords for the call thing_to_call(**d) """

    if 'self' in d.keys():
        raise TypeError('Your dictionary includes an entry for "self", which is just asking for trouble')

    thing_orig = thing_to_call

    if not isinstance(thing_to_call, types.FunctionType):
        if hasattr(thing_to_call,'__init__'):
            thing_to_call = thing_to_call.__init__
        elif hasattr(thing_to_call,'__call__'):
            thing_to_call = thing_to_call.__call__

    args, varargs, keywords, defaults = inspect.getargspec(thing_to_call)

    for arg in args:
        if not isinstance(arg,str):
            raise TypeError(str(thing_orig)+' uses tuple arguments, this is deprecated and not supported by the framework')

    if varargs != None:
        raise TypeError(str(thing_orig)+' has a variable length argument, but this is not supported by config resolution')

    if keywords is None:
        bad_keywords = [ arg_name for arg_name in d.keys() if arg_name not in args ]

        if len(bad_keywords) > 0:
            raise TypeError(str(thing_orig)+' does not support the following keywords: '+str(bad_keywords))

    if defaults is None:
        num_defaults = 0
    else:
        num_defaults = len(defaults)

    missing_keywords = [ arg for arg in args[0:len(args)-num_defaults] if arg not in d.keys() ]

    if len(missing_keywords) > 0:
        #iff the im_self field is present, this is a bound method, which has 'self' listed as an argument, but which should not be supplied by d
        if len(missing_keywords) > 1 or missing_keywords[0] != 'self' or not hasattr(thing_to_call,'im_self'):
            raise TypeError(str(thing_orig)+' expects the following arguments which were not present: '+str(missing_keywords))

def checked_call(thing_to_call, d):
    """ calls thing_to_call(**d) and raises a more-informative-than-usual exception if d does not contain the right arguments """
    try:
        return thing_to_call(**d)
    except TypeError:
        check(thing_to_call,d)
        raise

def resolve_model(d):
    assert False

def resolve_dataset(d):
    import framework.datasets.config
    return framework.datasets.config.resolve(d)

def resolve_train_algorithm(d):
    assert False

resolvers = {
        'model'             : resolve_model,
        'dataset'           : resolve_dataset,
        'train_algorithm'   : resolve_train_algorithm
        }


