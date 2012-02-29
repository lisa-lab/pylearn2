


class Dataset(object):

    def get_batch_design(self, batch_size, include_labels=False):
        """ Returns a randomly chosen batch of data formatted as a design matrix. """

        raise NotImplementedError()


    def get_batch_topo(self, batch_size):
        """ Returns a topology-preserving batch of data.
            The first index is over different examples, and has length batch_size.
            The next indices are the topologically significant dimensions of the
            data, i.e. for images, image rows followed by image columns.
            The last index is over separate channels.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    def set_iteration_scheme(self, mode=None, batch_size=None,
                             num_batches=None, topo=False):
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
        """
        raise NotImplementedError()

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, rng=None):
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
            The size of an individual batch. Unnecessary if `mode` is
            'sequential' and `num_batches` is specified.
        num_batches : int, optional
            The size of an individual batch. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified.
        topo : boolean, optional
            Whether batches returned by the iterator should present
            examples in a topological view or not. Defaults to
            `False`.

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).
        """
        raise NotImplementedError()
