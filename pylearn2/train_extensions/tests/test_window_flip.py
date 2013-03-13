import hashlib
import numpy
from pylearn2.train_extensions.window_flip import WindowAndFlipC01B

from pylearn2.datasets.dense_design_matrix import (
    DenseDesignMatrix,
    DefaultViewConverter
)
from pylearn2.utils.testing import assert_equal, assert_contains, assert_

class DummyDataset(DenseDesignMatrix):
    def __init__(self):
        axes = ['c', 0, 1, 'b']
        vc = DefaultViewConverter((5, 5, 2), axes=axes)
        rng = numpy.random.RandomState([2013, 3, 12])
        X = rng.normal(size=(4, 50)).astype('float32')
        super(DummyDataset, self).__init__(X=X, view_converter=vc, axes=axes)


def _hash_array(arr):
    h = hashlib.sha1(arr.copy())
    return h.hexdigest()


def test_window_flip_coverage():
    ddata = DummyDataset()
    topo = ddata.get_topological_view()
    ref_win = [set() for _ in xrange(4)]
    for b in xrange(topo.shape[-1]):
        for i in xrange(3):
            for j in xrange(3):
                window = topo[:, i:i + 3, j:j + 3, b]
                assert_equal((3, 3), window.shape[1:])
                ref_win[b].add(_hash_array(window))
                ref_win[b].add(_hash_array(window[:, :, ::-1]))
    actual_win = [set() for _ in xrange(4)]
    wf = WindowAndFlipC01B(window_shape=(3, 3))
    wf.setup(None, ddata, None)
    curr_topo = ddata.get_topological_view()
    assert_equal((2, 3, 3, 4), curr_topo.shape)
    for b in xrange(topo.shape[-1]):
        hashed = _hash_array(curr_topo[..., b])
        assert_contains(ref_win[b], hashed)
        actual_win[b].add(hashed)
    while not all(len(a) == len(b) for a, b in zip(ref_win, actual_win)):
        prev_topo = curr_topo.copy()
        wf.on_monitor(None, ddata, None)
        curr_topo = ddata.get_topological_view()
        assert_(not (prev_topo == curr_topo).all())
        for b in xrange(topo.shape[-1]):
            hashed = _hash_array(curr_topo[..., b])
            assert hashed in ref_win[b]
            actual_win[b].add(hashed)
