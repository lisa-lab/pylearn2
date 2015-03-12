"""
Objects for datasets serialized in HDF5 format (.h5).
"""

__author__ = "Francesco Visin"
__license__ = "3-clause BSD"
__credits__ = "Francesco Visin and Steven Kearnes"
__maintainer__ = "Francesco Visin"

try:
    import h5py
except ImportError:
    h5py = None
try:
    import tables
except ImportError:
    tables = None
import warnings
from os.path import isfile
from pylearn2.compat import OrderedDict
from pylearn2.datasets import cache
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.hdf5_deprecated import HDF5DatasetDeprecated
from pylearn2.utils import safe_zip, wraps, py_integer_types
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.exc import reraise_as
from pylearn2.space import Space, CompositeSpace
from theano.compat.six import string_types


class HDF5Dataset(Dataset):

    """
    Dataset loaded from an HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file name.
    sources : list of str
        A list of key(s) of dataset data in the HDF5 file.
    spaces : list of Space objects
        A list of spaces, one for each source in sources.
    aliases : list of str, optional
        A list of aliases, one for each source. They work as an alternative
        name for the sources of the dataset and if set can be used to access
        the data. If you use this parameter, for each element in sources
        you have to either specify an alias or None. Note that this allows you
        to use the new interface with legacy code by just setting the aliases
        `features` and, if needed, `targets`.
    load_all : bool, optional (default False)
        If true, datasets are loaded into memory instead of being left
        on disk.
    cache_size: int, optional
        This value is used when use_h5py is True.
        The size in bytes for the chunk cache of the HDF5 library. Useful
        when the HDF5 files has large chunks and when using a sequential
        iterator. The chunk cache allows only access the disk for the
        chunks and then copy the batches to GPU from memory, which can
        result in a significant speed up. Sensible default values depend
        on the size of your data and the batch size you wish to use. A
        rule of thumb is to make a chunk contain 100 - 1000 batches and
        make sure they encompass complete samples.
    use_h5py: bool or 'auto', optional
        Specifies if h5py or pytables should be used. If set to auto
        pylearn2 will try to use pytables and will switch to h5py if
        pytables cannot be loaded (e.g. is not installed).
    kwargs : dict, optional
        Keyword arguments passed to `DenseDesignMatrix`.
    """
    def __new__(cls, filename, X=None, topo_view=None, y=None, load_all=False,
                cache_size=None, sources=None, spaces=None, aliases=None,
                use_h5py='auto', **kwargs):
        """
        Temporary method to manage the deprecation
        """
        if X is not None or topo_view is not None:
            warnings.warn(
                'A dataset is using the old interface that is now deprecated '
                'and will become officially unsupported as of July 27, 2015. '
                'The dataset should use the new interface that inherits from '
                'the dataset class instead of the DenseDesignMatrix class. '
                'Please refer to pylearn2.datasets.hdf5.py for more details '
                'on arguments and details of the new '
                'interface.', DeprecationWarning)
            return HDF5DatasetDeprecated(filename, X, topo_view, y, load_all,
                                         cache_size, **kwargs)
        else:
            return super(HDF5Dataset, cls).__new__(
                cls, filename, sources, spaces, aliases, load_all, cache_size,
                use_h5py, **kwargs)

    def __init__(self, filename, sources, spaces, aliases=None, load_all=False,
                 cache_size=None, use_h5py='auto', **kwargs):
        """
        Class constructor
        """
        assert isinstance(filename, string_types)
        assert isfile(filename), '%s does not exist.' % filename
        assert isinstance(sources, list)
        assert all([isinstance(el, string_types) for el in sources])
        assert isinstance(spaces, list)
        assert all([isinstance(el, Space) for el in spaces])
        assert len(sources) == len(spaces)
        if aliases is not None:
            assert isinstance(aliases, list)
            assert all([isinstance(el, string_types) for el in aliases
                       if el is not None])
            assert len(aliases) == len(sources)
        assert isinstance(load_all, bool)
        assert cache_size is None or isinstance(cache_size, py_integer_types)
        assert isinstance(use_h5py, bool) or use_h5py == 'auto'

        self.load_all = load_all
        self._aliases = aliases if aliases else [None for _ in sources]
        self._sources = sources

        # self.spaces is a dictionary indexed with both keys and aliases
        self.spaces = alias_dict()
        for i, (source, alias) in enumerate(safe_zip(self._sources,
                                            self._aliases)):
            self.spaces[source, alias] = spaces[i]
        del spaces, aliases, sources

        if load_all:
            warnings.warn('You can load all the data in memory for speed, but '
                          'DO NOT use modify all the dataset at once (e.g., '
                          'reshape, transform, etc, ...) because your code '
                          'will fail if at some point you won\'t have enough '
                          'memory to store the dataset alltogheter. Use the '
                          'iterator to reshape the data after you load it '
                          'from the dataset.')

        # Locally cache the files before reading them
        datasetCache = cache.datasetCache
        filename = datasetCache.cache_file(filename)

        if use_h5py == 'auto':
            use_h5py = True if tables is None else False

        if use_h5py:
            if h5py is None:
                raise RuntimeError("Could not import h5py.")
            if cache_size:
                propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
                settings = list(propfaid.get_cache())
                settings[2] = cache_size
                propfaid.set_cache(*settings)
                self._fhandler = h5py.File(h5py.h5f.open(filename,
                                           fapl=propfaid), mode='r')
            else:
                self._fhandler = h5py.File(filename, mode='r')
        else:
            if tables is None:
                raise RuntimeError("Could not import tables.")
            self._fhandler = tables.openFile(filename, mode='r')

        self.data = self._read_hdf5(self._sources, self._aliases, load_all,
                                    use_h5py)

        assert len(self.data) != 0, (
            'No dataset was loaded. Please make sure that sources is a list '
            'with at least one value and that the provided values are keys of '
            'the dataset you are trying to load.')
        super(HDF5Dataset, self).__init__(**kwargs)

    def _read_hdf5(self, sources, aliases, load_all=False, use_h5py=True):
        """
        Loads elements from an HDF5 dataset using either h5py or tables. It can
        load either the whole object in memory or a reference to the object on
        disk, depending on the load_all parameter. Returns a list of objects.

        Parameters
        ----------
        sources : list of str
            List of HDF5 keys corresponding to the data to be loaded.
        load_all : bool, optional (default False)
            If true, load dataset into memory.
        use_h5py: bool, optional (default True)
            If true uses h5py, else tables.
        """
        data = alias_dict()
        if use_h5py:
            for s, a in safe_zip(sources, aliases):
                if load_all:
                    data[s, a] = self._fhandler[s][:]
                else:
                    data[s, a] = self._fhandler[s]
                    # hdf5 handle has no ndim
                    data[s].ndim = len(data[s].shape)
        else:
            for s, a in safe_zip(sources, aliases):
                if load_all:
                    data[s, a](self._fhandler.getNode('/', s)[:])
                else:
                    data[s, a] = self._fhandler.getNode('/', s)
        return data

    @wraps(Dataset.iterator, assigned=(), updated=(), append=True)
    def iterator(self, mode=None, data_specs=None, batch_size=None,
                 num_batches=None, rng=None, return_tuple=False, **kwargs):
        """
        if data_specs is set to None, the aliases (or sources) and spaces
        provided when the dataset object has been created will be used.
        """
        if data_specs is None:
            data_specs = (self._get_sources, self._get_spaces)

        [mode, batch_size, num_batches, rng, data_specs] = self._init_iterator(
            mode, batch_size, num_batches, rng, data_specs)
        convert = None

        return FiniteDatasetIterator(self,
                                     mode(self.get_num_examples(),
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def _get_sources(self):
        """
        Returns the aliases (if defined, sources otherwise) provided when the
        HDF5 object was created

        Returns
        -------
        A string or a list of strings.
        """
        return tuple([alias if alias else source for alias, source in
                      safe_zip(self._aliases, self._sources)])

    def _get_spaces(self):
        """
        Returns the Space(s) associated with the aliases (or sources) specified
        when the HDF5 object has been created.

        Returns
        -------
        A Space or a list of Spaces.
        """
        space = [self.spaces[s] for s in self._get_sources]
        return space[0] if len(space) == 1 else tuple(space)

    def get_data_specs(self, source_or_alias=None):
        """
            Returns a tuple `(space, source)` for each one of the provided
            source_or_alias keys, if any. If no key is provided, it will
            use self.aliases, if not None, or self.sources.
        """
        if source_or_alias is None:
            source_or_alias = self._get_sources()

        if isinstance(source_or_alias, (list, tuple)):
            space = tuple([self.spaces[s] for s in source_or_alias])
            space = CompositeSpace(space)
        else:
            space = self.spaces[source_or_alias]
        return (space, source_or_alias)

    def get_data(self):
        """
        DEPRECATED
        Returns all the data, as it is internally stored.
        The definition and format of these data are described in
        `self.get_data_specs()`.

        Returns
        -------
        data : numpy matrix or 2-tuple of matrices
            The data
        """
        return tuple([self.data[s] for s in self._get_sources()])

    def get(self, sources, indexes):
        """
        Retrieves the requested elements from the dataset.

        Parameter
        ---------
        sources : tuple
            A tuple of source identifiers
        indexes : slice or list
            A slice or a list of indexes

        Return
        ------
        rval : tuple
            A tuple of batches, one for each source
        """
        assert isinstance(sources, (tuple, list)) and len(sources) > 0, (
            'sources should be an instance of tuple and not empty')
        assert all([isinstance(el, string_types) for el in sources]), (
            'sources elements should be strings')
        assert isinstance(indexes, (tuple, list, slice, py_integer_types)), (
            'indexes should be either an int, a slice or a tuple/list of ints')
        if isinstance(indexes, (tuple, list)):
            assert len(indexes) > 0 and all([isinstance(i, py_integer_types)
                                            for i in indexes]), (
                'indexes elements should be ints')

        rval = []
        for s in sources:
            try:
                sdata = self.data[s]
            except ValueError as e:
                reraise_as(ValueError(
                    'The requested source %s is not part of the dataset' %
                    sources[s], *e.args))
            if (isinstance(indexes, (slice, py_integer_types)) or
                    len(indexes) == 1):
                rval.append(sdata[indexes])
            else:
                warnings.warn('Accessing non sequential elements of an '
                              'HDF5 file will be at best VERY slow. Avoid '
                              'using iteration schemes that access '
                              'random/shuffled data with hdf5 datasets!!')
                val = []
                [val.append(sdata[idx]) for idx in indexes]
                rval.append(val)
        return tuple(rval)

    @wraps(Dataset.get_num_examples, assigned=(), updated=())
    def get_num_examples(self, source_or_alias=None):
        """
        Return the number of examples *OF THE FIRST SOURCE*.
        Note that this behavior will probably be deprecated in the future,
        returing a list of num_examples. Do not rely on this function unless
        unavoidable.

        Parameter
        ---------
        source_or_alias : str, optional
            The source you want the number of examples of
        """
        assert source_or_alias is None or isinstance(source_or_alias,
                                                     string_types)

        if source_or_alias is None:
            alias = self._get_sources()
            alias = alias[0] if isinstance(alias, (list, tuple)) else alias
            data = self.data[alias]
        else:
            data = self.data[source_or_alias]

        return data.shape[0]


class alias_dict(OrderedDict):
    """
    A class that behaves like a dictionary, but let you associates a key and
    an alias to a value.
    IMPORTANT: Do not rely too much on this class, many functionalities are
    missing (e.g. element removal)
    """
    def __init__(self, **kwargs):
        self.__a2k__ = {}
        self.__k2a__ = {}
        super(alias_dict, self).__init__(**kwargs)

    def __getitem__(self, key_or_alias):
        """
        Returns the item corresponding to a key or an alias.

        Parameter
        ---------
        key_or_alias: any valid key for a dictionary
            A key or an alias.
        """
        assert isinstance(key_or_alias, string_types)
        try:
            return super(alias_dict, self).__getitem__(key_or_alias)
        except KeyError:
            return super(alias_dict, self).__getitem__(
                self.__a2k__[key_or_alias])

    def __setitem__(self, keys, value):
        """
        Add an element to the dictionary

        Parameter
        ---------
        keys: either a tuple `(key, alias)` or any valid key for a dictionary
            The key and optionally the alias of the new element.
        value: any input accepted as value by a dictionary
            The value of the new element.i

        Notes
        -----
        You can add elements to the dictionary as follows:
            1) my_dict[key] = value
            2) my_dict[key, alias] = value
        """
        assert isinstance(keys, (list, tuple, string_types))
        if isinstance(keys, (list, tuple)):
            assert all([el is None or isinstance(el, string_types)
                        for el in keys])

        if isinstance(keys, (list, tuple)):
            if keys[1] is not None:
                # workaround to avoid using the deprecated method has_key()
                if keys[0] in self.__a2k__ or keys[0] in super(alias_dict,
                                                               self).keys():
                    raise Exception('The key is already used in the '
                                    'dictionary either as key or alias')
                # workaround to avoid using the deprecated method has_key()
                if keys[1] in self.__a2k__ or keys[1] in super(alias_dict,
                                                               self).keys():
                    raise Exception('The alias is already used in the '
                                    'dictionary either as key or alias')
                self.__k2a__[keys[0]] = keys[1]
                self.__a2k__[keys[1]] = keys[0]
            keys = keys[0]
        return super(alias_dict, self).__setitem__(keys, value)

    def set_alias(self, key, alias):
        """
        Add an alias to a key of the dictionary that doesn't have already an
        alias.

        Parameter
        ---------
        keys: any valid key for a dictionary
           A key of the dictionary.
        alias: any input accepted as key by a dictionary
            The alias.
        """
        if alias is None:
            return
        # workaround to avoid using the deprecated method has_key()
        if key not in super(alias_dict, self).keys():
            raise NameError('The key is not in the dictionary')
        if key in self.__k2a__ and alias != self.__k2a__[key]:
            raise NameError('The key is already associated to a different '
                            'alias')
        # workaround to avoid using the deprecated method has_key()
        if (alias in self.__a2k__ and key != self.__a2k__[alias] or
                alias in super(alias_dict, self).keys()):
            raise Exception('The alias is already used in the dictionary '
                            'either as key or alias')
        self.__k2a__[key] = alias
        self.__a2k__[alias] = key

    def __contains__(self, key_or_alias):
        """
        Returns true if the key or alias is an element of the dictionary

        Parameter
        ---------
        keys_or_alias: any valid key for a dictionary
            The key or the alias to look for.
        """
        try:
            isalias = super(alias_dict, self).__contains__(
                self.__k2a__[key_or_alias])
        except KeyError:
            isalias = False
            pass
        return isalias or super(alias_dict, self).__contains__(key_or_alias)
