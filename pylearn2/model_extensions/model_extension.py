"""
Base class for model extensions
"""


class ModelExtension(object):
    """
    An object that may be plugged into a model to add some functionality
    to it.
    """

    def post_modify_updates(self, updates):
        """"
        Modifies the parameters before a learning update is applied.
        This method acts *after* the model subclass' _modify_updates
        method and any ModelExtensions that come earlier in the
        extensions list.

        Parameters
        ----------
        updates : dict
            A dictionary mapping shared variables to symbolic values they
            will be updated to.
        """

        pass
