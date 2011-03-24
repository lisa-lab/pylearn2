"""Support code for setting up YAML parsing."""
import yaml
from ..utils.call_check import checked_call

is_initialized = False

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
        classname = eval(tag_suffix)
        return checked_call(classname, mapping)

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
