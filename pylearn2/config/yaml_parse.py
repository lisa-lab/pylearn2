"""Support code for YAML parsing of experiment descriptions."""
import yaml
from pylearn2.utils import serial
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.call_check import checked_call
from pylearn2.utils.string_utils import match
from collections import namedtuple
import logging
import warnings
import re

from theano.compat import six

SCIENTIFIC_NOTATION_REGEXP = r'^[\-\+]?(\d+\.?\d*|\d*\.?\d+)?[eE][\-\+]?\d+$'

is_initialized = False
additional_environ = None
logger = logging.getLogger(__name__)

# Lightweight container for initial YAML evaluation.
#
# This is intended as a robust, forward-compatible intermediate representation
# for either internal consumption or external consumption by another tool e.g.
# hyperopt.
#
# We've included a slot for positionals just in case, though they are
# unsupported by the instantiation mechanism as yet.
BaseProxy = namedtuple('BaseProxy', ['callable', 'positionals',
                                     'keywords', 'yaml_src'])


class Proxy(BaseProxy):
    """
    An intermediate representation between initial YAML parse and object
    instantiation.

    Parameters
    ----------
    callable : callable
        The function/class to call to instantiate this node.
    positionals : iterable
        Placeholder for future support for positional arguments (`*args`).
    keywords : dict-like
        A mapping from keywords to arguments (`**kwargs`), which may be
        `Proxy`s or `Proxy`s nested inside `dict` or `list` instances.
        Keys must be strings that are valid Python variable names.
    yaml_src : str
        The YAML source that created this node, if available.

    Notes
    -----
    This is intended as a robust, forward-compatible intermediate
    representation for either internal consumption or external consumption
    by another tool e.g. hyperopt.

    This particular class mainly exists to  override `BaseProxy`'s `__hash__`
    (to avoid hashing unhashable namedtuple elements).
    """
    __slots__ = []

    def __hash__(self):
        """
        Return a hash based on the object ID (to avoid hashing unhashable
        namedtuple elements).
        """
        return hash(id(self))


def do_not_recurse(value):
    """
    Function symbol used for wrapping an unpickled object (which should
    not be recursively expanded). This is recognized and respected by the
    instantiation parser. Implementationally, no-op (returns the value
    passed in as an argument).

    Parameters
    ----------
    value : object
        The value to be returned.

    Returns
    -------
    value : object
        The same object passed in as an argument.
    """
    return value


def _instantiate_proxy_tuple(proxy, bindings=None):
    """
    Helper function for `_instantiate` that handles objects of the `Proxy`
    class.

    Parameters
    ----------
    proxy : Proxy object
        A `Proxy` object that.
    bindings : dict, opitonal
        A dictionary mapping previously instantiated `Proxy` objects
        to their instantiated values.

    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.
    """
    if proxy in bindings:
        return bindings[proxy]
    else:
        # Respect do_not_recurse by just un-packing it (same as calling).
        if proxy.callable == do_not_recurse:
            obj = proxy.keywords['value']
        else:
            # TODO: add (requested) support for positionals (needs to be added
            # to checked_call also).
            if len(proxy.positionals) > 0:
                raise NotImplementedError('positional arguments not yet '
                                          'supported in proxy instantiation')
            kwargs = dict((k, _instantiate(v, bindings))
                          for k, v in six.iteritems(proxy.keywords))
            obj = checked_call(proxy.callable, kwargs)
        try:
            obj.yaml_src = proxy.yaml_src
        except AttributeError:  # Some classes won't allow this.
            pass
        bindings[proxy] = obj
        return bindings[proxy]


def _instantiate(proxy, bindings=None):
    """
    Instantiate a (hierarchy of) Proxy object(s).

    Parameters
    ----------
    proxy : object
        A `Proxy` object or list/dict/literal. Strings are run through
        `preprocess`.
    bindings : dict, opitonal
        A dictionary mapping previously instantiated `Proxy` objects
        to their instantiated values.

    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.

    Notes
    -----
    This should not be considered part of the stable, public API.
    """
    if bindings is None:
        bindings = {}
    if isinstance(proxy, Proxy):
        return _instantiate_proxy_tuple(proxy, bindings)
    elif isinstance(proxy, dict):
        # Recurse on the keys too, for backward compatibility.
        # Is the key instantiation feature ever actually used, by anyone?
        return dict((_instantiate(k, bindings), _instantiate(v, bindings))
                    for k, v in six.iteritems(proxy))
    elif isinstance(proxy, list):
        return [_instantiate(v, bindings) for v in proxy]
    # In the future it might be good to consider a dict argument that provides
    # a type->callable mapping for arbitrary transformations like this.
    elif isinstance(proxy, six.string_types):
        return preprocess(proxy)
    else:
        return proxy


def load(stream, environ=None, instantiate=True, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the .read() interface.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.
    instantiate : bool, optional
        If `False`, do not actually instantiate the objects but instead
        produce a nested hierarchy of `Proxy` objects.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `Proxy` objects.

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    global is_initialized
    global additional_environ
    if not is_initialized:
        initialize()
    additional_environ = environ

    if isinstance(stream, six.string_types):
        string = stream
    else:
        string = stream.read()

    proxy_graph = yaml.load(string, **kwargs)
    if instantiate:
        return _instantiate(proxy_graph)
    else:
        return proxy_graph


def load_path(path, environ=None, instantiate=True, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.
    instantiate : bool, optional
        If `False`, do not actually instantiate the objects but instead
        produce a nested hierarchy of `Proxy` objects.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `Proxy` objects.

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    with open(path, 'r') as f:
        content = ''.join(f.readlines())

    # This is apparently here to avoid the odd instance where a file gets
    # loaded as Unicode instead (see 03f238c6d). It's rare instance where
    # basestring is not the right call.
    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str, got " +
                             str(type(content)))

    return load(content, instantiate=instantiate, environ=environ, **kwargs)


def try_to_import(tag_suffix):
    """
    .. todo::

        WRITEME
    """
    components = tag_suffix.split('.')
    modulename = '.'.join(components[:-1])
    try:
        exec('import %s' % modulename)
    except ImportError as e:
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
            reraise_as(ImportError("Could not import %s; ImportError was %s" %
                                   (modulename, str_e)))
        else:

            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                modulename = '.'.join(pcomponents[:j])
                try:
                    exec('import %s' % modulename)
                except Exception:
                    base_msg = 'Could not import %s' % modulename
                    if j > 1:
                        modulename = '.'.join(pcomponents[:j - 1])
                        base_msg += ' but could import %s' % modulename
                    reraise_as(ImportError(base_msg + '. Original exception: '
                                           + str(e)))
                j += 1
    try:
        obj = eval(tag_suffix)
    except AttributeError as e:
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

        except Exception:
            warnings.warn("Attempt to decipher AttributeError failed")
            reraise_as(AttributeError('Could not evaluate %s. ' % tag_suffix +
                                      'Original error was ' + str(e)))
        reraise_as(AttributeError(msg))
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

    pattern = re.compile(SCIENTIFIC_NOTATION_REGEXP)
    yaml.add_implicit_resolver('!float', pattern)

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
        if not isinstance(key, six.string_types):
            message = "Received non string object (%s) as " \
                      "key in mapping." % str(key)
            raise TypeError(message)
    if '.' not in tag_suffix:
        # TODO: I'm not sure how this was ever working without eval().
        callable = eval(tag_suffix)
    else:
        callable = try_to_import(tag_suffix)
    rval = Proxy(callable=callable, yaml_src=yaml_src, positionals=(),
                 keywords=mapping)
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
    obj = serial.load(preprocess(mapping, additional_environ))
    proxy = Proxy(callable=do_not_recurse, positionals=(),
                  keywords={'value': obj}, yaml_src=yaml.serialize(node))
    return proxy


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
        except TypeError as exc:
            const = yaml.constructor
            reraise_as(const.ConstructorError("while constructing a mapping",
                                              node.start_mark,
                                              "found unacceptable key (%s)" %
                                              (exc, key_node.start_mark)))
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
