from pylearn2.datasets.mnist import MNIST
train = MNIST(which_set = 'train')
test = MNIST(which_set = 'test')

def test_range():
    """Tests that the data spans [0,1]"""
    for X in [train.X, test.X ]:
        assert X.min() == 0.0
        assert X.max() == 1.0

def test_topo():
    """Tests that a topological batch has 4 dimensions"""
    topo = train.get_batch_topo(1)
    assert topo.ndim == 4
