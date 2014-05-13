"""Support code for YAML parsing of experiment descriptions."""
import re
import yaml
from pylearn2.utils.call_check import checked_call
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.string_utils import match
import logging
import warnings
import collections


is_initialized = False
additional_environ = None
logger = logging.getLogger(__name__)



def load(stream, overrides=None, environ=None, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object \
        supporting the .read() interface.
    overrides : dict, optional [DEPRECATED]
        A dictionary containing overrides to apply. The location of \
        the override is specified in the key as a dot-delimited path \
        to the desired parameter, e.g. "model.corruptor.corruption_level".
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    global is_initialized
    global additional_environ
    if not is_initialized:
        initialize()
    additional_environ = environ

    if isinstance(stream, basestring):
        string = stream
    else:
        string = '\n'.join(stream.readlines())

    proxy_graph = yaml.load(string, **kwargs)

    if overrides is not None:
        warnings.warn("The 'overrides' keyword is deprecated and will "
                      "be removed on or after June 8, 2014.")
        handle_overrides(proxy_graph, overrides)
    return instantiate_all(proxy_graph)


def load_path(path, overrides=None, environ=None, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of \
        the override is specified in the key as a dot-delimited path \
        to the desired parameter, e.g. "model.corruptor.corruption_level".
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified a \
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    f = open(path, 'r')
    content = ''.join(f.readlines())
    f.close()

    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str, got " +
                             str(type(content)))

    return load(content, environ=environ, **kwargs)


def handle_overrides(graph, overrides):
    """
    Handle any overrides for this model configuration.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph \
        loaded from a YAML file.
    overrides : dict
        A dictionary containing overrides to apply. The location of \
        the override is specified in the key as a dot-delimited path \
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
        A dictionary (or an ObjectProxy) containing the object graph \
        loaded from a YAML file.

    Returns
    -------
    graph : dict or object
        The dictionary or object resulting after the recursive instantiation.
    """

    def should_instantiate(obj):
        classes = (ObjectProxy, dict, list)
        return isinstance(obj, classes)

    if not isinstance(graph, list):
        for key in graph:
            if should_instantiate(graph[key]):
                graph[key] = instantiate_all(graph[key])
            if isinstance(graph[key], basestring):       # preprocess strings
                graph[key] = preprocess(graph[key], additional_environ)

        if hasattr(graph, 'keys'):
            for key in graph.keys():
                if should_instantiate(key):
                    new_key = instantiate_all(key)
                    graph[new_key] = graph[key]
                    del graph[key]

    if isinstance(graph, ObjectProxy):
        graph = graph.instantiate()

    if isinstance(graph, list):
        for i, elem in enumerate(graph):
            if should_instantiate(elem):
                graph[i] = instantiate_all(elem)

    return graph


class ObjectProxy(object):
    """
    Class used to delay instantiation of objects so that overrides can be
    applied.

    Parameters
    ----------
    cls : WRITEME
    kwds : WRITEME
    yaml_src : WRITEME
    """
    def __init__(self, cls, kwds, yaml_src):
        self.cls = cls
        self.kwds = kwds
        self.yaml_src = yaml_src
        self.instance = None

    def __setitem__(self, key, value):
        """
        .. todo::

            WRITEME
        """
        self.kwds[key] = value

    def __getitem__(self, key):
        """
        .. todo::

            WRITEME
        """
        return self.kwds[key]

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self.kwds.__iter__()

    def keys(self):
        """
        .. todo::

            WRITEME
        """
        return list(self.kwds)

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


def try_to_import(tag_suffix):
    """
    .. todo::

        WRITEME
    """
    components = tag_suffix.split('.')
    modulename = '.'.join(components[:-1])
    try:
        exec('import %s' % modulename)
    except ImportError, e:
        # We know it's an ImportError, but is it an ImportError related to
        # this path,
        # or did the module we're importing have an unrelated ImportError?
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

            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                modulename = '.'.join(pcomponents[:j])
                try:
                    exec('import %s' % modulename)
                except:
                    base_msg = 'Could not import %s' % modulename
                    if j > 1:
                        modulename = '.'.join(pcomponents[:j - 1])
                        base_msg += ' but could import %s' % modulename
                    raise ImportError(base_msg + '. Original exception: '
                                      + str(e))
                j += 1
    try:
        obj = eval(tag_suffix)
    except AttributeError, e:
        try:
            # Try to figure out what the wrong field name was
            # If we fail to do it, just fall back to giving the usual
            # attribute error
            pieces = tag_suffix.split('.')
            module = '.'.join(pieces[:-1])
            field = pieces[-1]
            candidates = dir(eval(module))

            msg = ('Could not evaluate %s. ' % tag_suffix +
                   'Did you mean ' + match(field, candidates) + '? ' +
                   'Original error was ' + str(e))

        except:
            warnings.warn("Attempt to decipher AttributeError failed")
            raise AttributeError('Could not evaluate %s. ' % tag_suffix +
                                 'Original error was ' + str(e))
        raise AttributeError(msg)
    return obj


def initialize():
    """
    Initialize the configuration system by installing YAML handlers.
    Automatically done on first call to load() specified in this file.
    """
    global is_initialized

    # Add the custom multi-constructor
    yaml.add_multi_constructor('!obj:', multi_constructor_obj)
    yaml.add_multi_constructor('!pkl:', multi_constructor_pkl)
    yaml.add_multi_constructor('!import:', multi_constructor_import)

    yaml.add_constructor('!import', constructor_import)
    yaml.add_constructor("!float", constructor_float)

    is_initialized = True


###############################################################################
# Callbacks used by PyYAML

def multi_constructor_obj(loader, tag_suffix, node):
    """
    Callback used by PyYAML when a "!obj:" tag is encountered.

    See PyYAML documentation for details on the call signature.
    """
    yaml_src = yaml.serialize(node)
    construct_mapping(node)
    mapping = loader.construct_mapping(node)

    assert hasattr(mapping, 'keys')
    assert hasattr(mapping, 'values')

    for key in mapping.keys():
        if not isinstance(key, basestring):
            message = "Received non string object (%s) as " \
                      "key in mapping." % str(key)
            raise TypeError(message)

    if '.' not in tag_suffix:
        classname = tag_suffix
        rval = ObjectProxy(classname, mapping, yaml_src)
    else:
        classname = try_to_import(tag_suffix)
        rval = ObjectProxy(classname, mapping, yaml_src)

    return rval


def multi_constructor_pkl(loader, tag_suffix, node):
    """
    Callback used by PyYAML when a "!pkl:" tag is encountered.
    """
    global additional_environ
    if tag_suffix != "" and tag_suffix != u"":
        raise AssertionError('Expected tag_suffix to be "" but it is "'
                             + tag_suffix +
                             '": Put space between !pkl: and the filename.')

    mapping = loader.construct_yaml_str(node)
    rval = ObjectProxy(None, {}, yaml.serialize(node))
    rval.instance = serial.load(preprocess(mapping, additional_environ))

    return rval


def multi_constructor_import(loader, tag_suffix, node):
    """
    Callback used by PyYAML when a "!import:" tag is encountered.
    """
    if '.' not in tag_suffix:
        raise yaml.YAMLError("!import: tag suffix contains no '.'")
    return try_to_import(tag_suffix)


def constructor_import(loader, node):
    """
    Callback used by PyYAML when a "!import <str>" tag is encountered.
    This tag exects a (quoted) string as argument.
    """
    value = loader.construct_scalar(node)
    if '.' not in value:
        raise yaml.YAMLError("import tag suffix contains no '.'")
    return try_to_import(value)


def constructor_float(loader, node):
    """
    Callback used by PyYAML when a "!float <str>" tag is encountered.
    This tag exects a (quoted) string as argument.
    """
    value = loader.construct_scalar(node)
    return float(value)


def construct_mapping(node, deep=False):
    # This is a modified version of yaml.BaseConstructor.construct_mapping
    # in which a repeated key raises a ConstructorError
    if not isinstance(node, yaml.nodes.MappingNode):
        const = yaml.constructor
        message = "expected a mapping node, but found"
        raise const.ConstructorError(None, None,
                                     "%s %s " % (message, node.id),
                                     node.start_mark)
    mapping = {}
    constructor = yaml.constructor.BaseConstructor()
    for key_node, value_node in node.value:
        key = constructor.construct_object(key_node, deep=False)
        try:
            hash(key)
        except TypeError, exc:
            const = yaml.constructor
            raise const.ConstructorError("while constructing a mapping",
                                         node.start_mark,
                                         "found unacceptable key (%s)" % exc,
                                         key_node.start_mark)
        if key in mapping:
            const = yaml.constructor
            raise const.ConstructorError("while constructing a mapping",
                                         node.start_mark,
                                         "found duplicate key (%s)" % key)
        value = constructor.construct_object(value_node, deep=False)
        mapping[key] = value
    return mapping

if __name__ == "__main__":
    initialize()
    # Demonstration of how to specify objects, reference them
    # later in the configuration, etc.
    yamlfile = """{
        "corruptor" : !obj:pylearn2.corruption.GaussianCorruptor &corr {
            "corruption_level" : 0.9
        },
        "dae" : !obj:pylearn2.models.autoencoder.DenoisingAutoencoder {
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
    logger.info(loaded)
    # These two things should be the same object
    assert loaded['corruptor'] is loaded['dae'].corruptor
