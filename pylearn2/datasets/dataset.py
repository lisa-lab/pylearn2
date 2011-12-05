


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
