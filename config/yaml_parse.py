"""Support code for setting up YAML parsing."""
import yaml
from ..utils.call_check import checked_call

is_initialized = False

def load(stream, overrides=None, **kwargs):
    global is_initialized
    if not is_initialized:
        initialize()
    proxy_tree = yaml.load(stream, **kwargs)
    import pdb; pdb.set_trace()
    if overrides is not None:
        handle_overrides(proxy_tree, overrides)
    return instantiate_all(proxy_tree)

def load_path(path, **kwargs):
    f =  open( path, 'r')
    content = ''.join(f.readlines())
    f.close()
    return load(content, **kwargs)

def handle_overrides(tree, overrides):
    for key in overrides:
        levels = key.split('.')
        part = tree
        for lvl in levels[:-1]:
            try:
                part = part[lvl]
            except KeyError:
                raise KeyError("'%s' override failed at '%s'", (key, lvl))
        part[levels[-1]] = overrides[key]

def instantiate_all(tree):
    for key in tree:
        if isinstance(tree[key], ObjectProxy) or isinstance(tree[key], dict):
            tree[key] = instantiate_all(tree[key])
    if isinstance(tree, ObjectProxy):
        tree = tree.instantiate()
    return tree

class ObjectProxy(object):
    def __init__(self, cls, kwds):
        self.cls = cls
        self.kwds = kwds
        self.instance = None

    def __setitem__(self, key, value):
        self.kwds[key] = value

    def __getitem__(self, key):
        return self.kwds[key]

    def __iter__(self):
        return self.kwds.__iter__()

    def instantiate(self):
        if self.instance is None:
            self.instance = checked_call(self.cls, self.kwds)
        return self.instance

def multi_constructor(loader, tag_suffix, node) :
    """
    Constructor function passed to PyYAML telling it how to construct
    objects from argument descriptions. See PyYAML documentation for
    details on the call signature.
    """
    mapping = loader.construct_mapping(node)
    if '.' not in tag_suffix:
        classname = tag_suffix
        return checked_call(classname, mapping)
    else:
        components = tag_suffix.split('.')
        modulename = '.'.join(components[:-1])
        exec('import %s' % modulename)
        try:
            classname = eval(tag_suffix)
        except AttributeError:
            raise AttributeError('Could not evaluate %s' % tag_suffix)
        return ObjectProxy(classname, mapping)

def initialize():
    global is_initialized
    # Add the custom multi-constructor
    yaml.add_multi_constructor('!obj:', multi_constructor)
    is_initialized = True

if __name__ == "__main__":
    initialize()
    # Demonstration of how to specify objects, reference them
    # later in the configuration, etc.
    yamlfile = """{
        "corruptor" : !obj:framework.corruption.GaussianCorruptor &corr {
            "corruption_level" : 0.9
        },
        "dae" : !obj:framework.autoencoder.DenoisingAutoencoder {
            "nhid" : 20,
            "nvis" : 30,
            "act_enc" : null,
            "act_dec" : null,
            "tied_weights" : true,
            # we could have also just put the corruptor definition here
            "corruptor" : *corr
        }
    }"""
    # yaml.load can take a string or a file object
    loaded = yaml.load(yamlfile)
    print loaded
    # These two things should be the same object
    assert loaded['corruptor'] is loaded['dae'].corruptor
