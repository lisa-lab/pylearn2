from unittest import TestCase
from pylearn2.datasets.text.sentiment_treebank import StanfordSentimentTreebank


class TestSentimentTreebank(TestCase):

    def setUp(self):
        self.train = StanfordSentimentTreebank('train')
        self.valid = StanfordSentimentTreebank('valid')
        self.test = StanfordSentimentTreebank('test')

    def test_topo(self):
        assert self.train.X.shape == (8544, 21701)
        assert self.train.y.shape == (8544,)

        assert self.valid.X.shape == (1101, 21701)
        assert self.valid.y.shape == (1101,)

        assert self.test.X.shape == (2210, 21701)
        assert self.test.y.shape == (2210,)



