"""Support code for YAML parsing of experiment descriptions."""
import yaml
from pylearn2.utils.call_check import checked_call
from pylearn2.utils import serial
from pylearn2.utils.string import preprocess


is_initialized = False

def load(stream, overrides=None, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the .read() interface.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified an
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    global is_initialized
    if not is_initialized:
        initialize()

    if isinstance(stream,str):
        string = stream
    else:
        string = '\n'.join(stream.readlines())

    processed_string = preprocess(string)

    proxy_graph = yaml.load(processed_string, **kwargs)

    #import pdb; pdb.set_trace()
    if overrides is not None:
        handle_overrides(proxy_graph, overrides)
    return instantiate_all(proxy_graph)


def load_path(path, overrides=None, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified an
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    f = open(path, 'r')
    content = ''.join(f.readlines())
    f.close()
    return load(content, **kwargs)


def handle_overrides(graph, overrides):
    """
    Handle any overrides for this model configuration.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.
    overrides : dict
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".
    """
    for key in overrides:
        levels = key.split('.')
        part = graph
        for lvl in levels[:-1]:
            try:
                part = part[lvl]
            except KeyError:
                raise KeyError("'%s' override failed at '%s'", (key, lvl))
        try:
            part[levels[-1]] = overrides[key]
        except KeyError:
            raise KeyError("'%s' override failed at '%s'", (key, levels[-1]))


def instantiate_all(graph):
    """
    Instantiate all ObjectProxy objects in a nested hierarchy.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.

    Returns
    -------
    graph : dict or object
        The dictionary or object resulting after the recursive instantiation.
    """

    def should_instantiate(obj):
        classes = [ObjectProxy, dict, list]
        return True in [isinstance(obj, cls) for cls in classes]

    if not isinstance(graph, list):
        for key in graph:
            if should_instantiate(graph[key]):
                graph[key] = instantiate_all(graph[key])
            #endif
        #endfor
    #endif
    if isinstance(graph, ObjectProxy):
        graph = graph.instantiate()
    #endif
    if isinstance(graph, list):
        for i, elem in enumerate(graph):
            if should_instantiate(elem):
                graph[i] = instantiate_all(elem)
            #
        #endfor
    #endif

    return graph


class ObjectProxy(object):
    """
    Class used to delay instantiation of objects so that overrides can be
    applied.
    """
    def __init__(self, cls, kwds, yaml_src):
        """

        """
        self.cls = cls
        self.kwds = kwds
        self.yaml_src = yaml_src
        self.instance = None

    def __setitem__(self, key, value):
        self.kwds[key] = value

    def __getitem__(self, key):
        return self.kwds[key]

    def __iter__(self):
        return self.kwds.__iter__()

    def instantiate(self):
        """
        Instantiate this object with the supplied parameters in `self.kwds`,
        or if already instantiated, return the cached instance.
        """
        if self.instance is None:
            self.instance = checked_call(self.cls, self.kwds)
        #endif
        try:
            self.instance.yaml_src = self.yaml_src
        except AttributeError:
            pass
        return self.instance


def multi_constructor(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to construct
    objects from argument descriptions. See PyYAML documentation for
    details on the call signature.
    """

    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)
    if '.' not in tag_suffix:
        classname = tag_suffix
        rval = ObjectProxy(classname, mapping, yaml_src)
    else:
        components = tag_suffix.split('.')
        modulename = '.'.join(components[:-1])

        try:
            exec('import %s' % modulename)
        except ImportError, e:
            # We know it's an ImportError, but is it an ImportError related to
            # this path,
            #o r did the module we're importing have an unrelated ImportError?
            # and yes, this test can still have false positives, feel free to
            # improve it
            pieces = modulename.split('.')
            str_e = str(e)
            found = True in [piece.find(str(e)) != -1 for piece in pieces]

            if found:
                # The yaml file is probably to blame.
                # Report the problem with the full module path from the YAML
                # file
                raise ImportError("Could not import %s; ImportError was %s" %
                                  (modulename, str_e))
            else:
                # The module being imported contains an error.
                # Pass the original exception on up, with the original stack
                # trace preserved
                raise
        try:
            classname = eval(tag_suffix)
        except AttributeError:
            raise AttributeError('Could not evaluate %s' % tag_suffix)
        rval = ObjectProxy(classname, mapping, yaml_src)

    return rval


def multi_constructor_pkl(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to load
    objects from paths to .pkl files. See PyYAML documentation for
    details on the call signature.
    """

    #print dir(loader)
    mapping = loader.construct_yaml_str(node)
    assert tag_suffix == ""

    rval = ObjectProxy(None, {}, yaml.serialize(node))
    rval.instance = serial.load(mapping)

    return rval


def initialize():
    """
    Initialize the configuration system by installing YAML handlers.
    Automatically done on first call to load() specified in this file.
    """
    global is_initialized
    # Add the custom multi-constructor
    yaml.add_multi_constructor('!obj:', multi_constructor)
    yaml.add_multi_constructor('!pkl:', multi_constructor_pkl)
    is_initialized = True

if __name__ == "__main__":
    initialize()
    # Demonstration of how to specify objects, reference them
    # later in the configuration, etc.
    yamlfile = """{
        "corruptor" : !obj:pylearn2.corruption.GaussianCorruptor &corr {
            "corruption_level" : 0.9
        },
        "dae" : !obj:pylearn2.autoencoder.DenoisingAutoencoder {
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
