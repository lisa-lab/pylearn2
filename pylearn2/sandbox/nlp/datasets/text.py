"""Datasets for working with text"""


class TextDatasetMixin(object):
    """
    Use multiple inheritance with this class and any other dataset
    class in order to provide useful functionality for natural
    language processing purposes.

    The derived class is expected to have a `_vocabulary`, which
    is a dictionary from words (strings) to indices (integers). If
    needed, one can also set the `_unknown_index` and `_unknown word`
    attributes, which define the index and string that will be used
    when a word or word index is not in the (inverse) dictionary
    respectively.
    """
    @property
    def vocabulary(self):
        """
        Returns the vocabulary (a dictionary from
        word to word indices)
        """
        if hasattr(self, '_vocabulary'):
            return self._vocabulary
        else:
            raise NotImplementedError('No vocabulary given')

    @property
    def unknown_index(self):
        """
        The index referring to the unknown word.
        """
        if not hasattr(self, '_unknown_index') and \
                0 in self.inverse_vocabulary:
            raise NotImplementedError('This dataset does not define an index '
                                      'for unknown words, but the default `0` '
                                      'is already taken')
        return getattr(self, '_unknown_index', 0)

    @property
    def unknown_word(self):
        """
        The string to use for the unknown words. If
        not defined, return `UNK`.
        """
        if not hasattr(self, '_unknown_word') and 0 in self.inverse_vocabulary:
            raise NotImplementedError('This dataset does not define an index '
                                      'for unknown words, but the default `0` '
                                      'is already taken')
        return getattr(self, '_unknown_word', 'UNK')

    @property
    def inverse_vocabulary(self):
        """
        The inverse vocabulary, a dictionary from
        integers to strings. If it does not exist,
        it is created from the vocabulary if possible.
        """
        if hasattr(self, '_inverse_vocabulary'):
            return self._inverse_vocabulary
        elif hasattr(self, '_vocabulary'):
            self._inverse_vocabulary = {index: word for word, index
                                        in self._vocabulary.iteritems()}
            return self._inverse_vocabulary
        else:
            raise NotImplementedError

    def words_to_indices(self, words):
        """
        Converts text to word indices using a dictionary

        Parameters
        ----------
        words : str or list of strings
            If passed a string, it will assume a sentence that
            is split at any whitespace. If a list, it assumes
            each element is a word.
        """
        return [self.vocabulary.get(word, self.unknown_index)
                for word in words]

    def indices_to_words(self, indices):
        """
        Converts word indices back to words and returns
        a list of strings

        Parameters
        ----------
        indices : list of ints
            A list of word indices
        """
        return [self.inverse_vocabulary.get(index, self.unknown_word)
                for index in indices]
