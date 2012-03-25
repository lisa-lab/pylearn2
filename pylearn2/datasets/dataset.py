class Dataset(object):
    """Abstract interface for Datasets."""
    def get_batch_design(self, batch_size, include_labels=False):
        """
        Returns a randomly chosen batch of data formatted as a design
        matrix.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def get_batch_topo(self, batch_size):
        """
        Returns a topology-preserving batch of data.

        The first index is over different examples, and has length
        batch_size. The next indices are the topologically significant
        dimensions of the data, i.e. for images, image rows followed by
        image columns.  The last index is over separate channels.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    def set_iteration_scheme(self, mode=None, batch_size=None,
                             num_batches=None, topo=False, targets=False):
        """
        Modify the default iteration behaviour for the dataset.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
        batch_size : int, optional
            The size of an individual batch. Unnecessary if `mode` is
            'sequential' and `num_batches` is specified.
        num_batches : int, optional
            The size of an individual batch. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified.
        topo : boolean, optional
            Whether batches returned by the iterator should present
            examples in a topological view or not. Defaults to
            `False`.

        Notes
        -----
        This method modifies the behaviour when one iterates on a
        dataset as a container, e.g. "for batch in dataset". One
        can also override any subset of these parameters by calling
        the `iterator()` method directly to obtain an iterator
        with specified behaviour.
        """
        # TODO: the logic from DenseDesignMatrix.set_iteration_scheme
        # is potentially better here.
        raise NotImplementedError()

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=False, rng=None):
        """
        Return an iterator for this dataset with the specified
        behaviour. Unspecified values are filled-in by the default.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        topo : boolean, optional
            Whether batches returned by the iterator should present
            examples in a topological view or not. Defaults to
            `False`.
        rng : int, object or array_like, optional
            Either an instance of `numpy.random.RandomState` (or
            something with a compatible interface), or a seed value
            to be passed to the constructor to create a `RandomState`.
            See the docstring for `numpy.random.RandomState` for
            details on the accepted seed formats. If unspecified,
            defaults to using the dataset's own internal random
            number generator, which persists across iterations
            through the dataset and may potentially be shared by
            multiple iterator objects simultaneously (see "Notes"
            below).

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).

        Notes
        -----
        Arguments are passed as instantiation parameters to classes
        that derive from `pylearn2.utils.iteration.SubsetIterator`.

        Iterating simultaneously with multiple iterator objects
        sharing the same random number generator could lead to
        difficult-to-reproduce behaviour during training. It is
        therefore *strongly recommended* that each iterator be given
        its own random number generator with the `rng` parameter
        in such situations.
        """
        # TODO: See how much of the logic from DenseDesignMatrix.iterator
        # can be handled here.
        raise NotImplementedError()
