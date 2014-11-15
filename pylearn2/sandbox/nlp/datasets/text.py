"""Datasets for working with text"""
from theano.compat import six


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
    def is_case_sensitive(self):
        return getattr(self, '_is_case_sensitive', True)

    @property
    def vocabulary(self):
        """
        Returns the vocabulary (a dictionary from
        word to word indices)
        """
        if hasattr(self, '_vocabulary'):
            if not getattr(self, '_vocabulary_case_checked', False):
                for word in self._vocabulary:
                    if word != word.lower():
                        raise ValueError('The vocabulary contains cased words '
                                         '(%s) but the dataset is supposed to '
                                         'be case-insensitive' % (word))
                self._vocabulary_case_checked = True
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
        if not hasattr(self, '_unknown_word') and 'UNK' in self.vocabulary:
            raise NotImplementedError('This dataset does not define a string '
                                      'for unknown words, but the default '
                                      '`UNK` is already taken')
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
            self._inverse_vocabulary = dict((index, word) for word, index
                                            in six.iteritems(self._vocabulary))
            return self._inverse_vocabulary
        else:
            raise NotImplementedError

    def words_to_indices(self, words):
        """
        Converts the elements of a (nested) list of strings
        to word indices

        Parameters
        ----------
        words : (nested) list of strings
            Assumes each element is a word
        """
        assert isinstance(words, list)
        if all(isinstance(word, list) for word in words):
            return [self.words_to_indices(word) for word in words]
        assert all(isinstance(word, six.string_types) for word in words)
        if self.is_case_sensitive:
            return [self.vocabulary.get(word, self.unknown_index)
                    for word in words]
        else:
            return [self.vocabulary.get(word.lower(), self.unknown_index)
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
