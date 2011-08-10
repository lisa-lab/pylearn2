


class Dataset(object):
    #TODO: bring in methods from existing dataset classes that all datasets should implement
    #TODO: have existing dataset classes inherit from this one

    def get_batch_topo(self, batch_size):
        """ Returns a topology-preserving batch of data.
            The first index is over different examples, and has length batch_size.
            The next indices are the topologically significant dimensions of the
            data, i.e. for images, image rows followed by image columns.
            The last index is over separate channels.
        """
        raise NotImplementedError()
