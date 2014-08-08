from unittest import TestCase
from pylearn2.testing import skip
from pylearn2.sandbox.nlp.datasets.sentiment_treebank import StanfordSentimentTreebank


class TestSentimentTreebank(TestCase):

    def setUp(self):
        skip.skip_if_no_data()
        self.train = StanfordSentimentTreebank('train')
        self.valid = StanfordSentimentTreebank('valid')
        self.test = StanfordSentimentTreebank('test')

    def test_topo(self):
        skip.skip_if_no_data()
        assert self.train.X.shape == (8544, 21701)
        assert self.train.y.shape == (8544,)

        assert self.valid.X.shape == (1101, 21701)
        assert self.valid.y.shape == (1101,)

        assert self.test.X.shape == (2210, 21701)
        assert self.test.y.shape == (2210,)



