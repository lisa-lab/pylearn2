import hashlib
import itertools
import numpy
from pylearn2.train_extensions.window_flip import WindowAndFlip
from pylearn2.train_extensions.window_flip import WindowAndFlipC01B

from pylearn2.datasets.dense_design_matrix import (
    DenseDesignMatrix,
    DefaultViewConverter
)
from pylearn2.utils.testing import assert_equal, assert_contains, assert_


# def _make_design_matrix(layout_axes):
#     num_batches = 4
#     num_rows = 5
#     num_cols = 5
#     num_channels = 2
#     b01c_shape = (num_batches, num_rows, num_cols, num_channels)
#     b01c_axes = ('b', 0, 1, 'c')

#     layout_shape = [b01c_shape[b01c_axes.index(ax) for ax in layout_axes]]

#     # topo = numpy.zeros(layout_shape, dtype='float32')
#     # for b in range(num_batches):

#     datum_size = num_rows * num_cols * num_channels
#     # result = numpy.array([1000
#     size = numpy_prod(b01c_shape)
#     data = numpy.arange(size, dtype='float32')
#     data = data.reshape(layout_shape)
#     for d0 in range(b01c_shape[b01c_axes.index(layout_axes[0])]):
#         for d1 in range(b01c_shape[b01c_axes.index(layout_axes[1])]):
#             for d2 in range(b01c_shape[b01c_axes.index(layout_axes[2])]):
#                 for d3 in range(b01c_shape[b01c_axes.index(layout_axes[3])]):
#                     result[d0, d1, d2, d3] = d0 * 1000 + d1 * 100 + d2 * 10 + d3

#     return result.reshape((num_batches, row_size))

#     # layout_shape = numpy.zeros(4)
#     # layout_shape[layout_axes.index['b'] = num_batches
#     # b01c_shape = (num_batches, num_rows, num_cols, num_channels)
#     # layout_shape= (layout_axes.index(ax)


class DummyDataset(DenseDesignMatrix):
    def __init__(self, axes=('c', 0, 1, 'b')):
        assert_contains([('c', 0, 1, 'b'), ('b', 0, 1, 'c')], axes)
        axes = list(axes)
        vc = DefaultViewConverter((5, 5, 2), axes=axes)
        rng = numpy.random.RandomState([2013, 3, 12])
        #X = _make_design_matrix(layout_axes=('b', 0, 1, 'c'))
        X = rng.normal(size=(4, 50)).astype('float32')
        super(DummyDataset, self).__init__(X=X, view_converter=vc, axes=axes)


def _hash_array(arr):
    h = hashlib.sha1(arr.copy())
    return h.hexdigest()

# def _hash_array(arr):
#     return str(arr)


def test_window_flip_coverage():
    # Old interface WindowAndFlipC01B
    yield check_window_flip_coverage_C01B, True, True
    yield check_window_flip_coverage_C01B, False, True
    # New interface WindowAndFlip
    yield check_window_flip_coverage_C01B, True
    yield check_window_flip_coverage_C01B, False
    yield check_window_flip_coverage_B01C, True
    yield check_window_flip_coverage_B01C, False


def check_window_flip_coverage_C01B(flip, use_old_c01b=False):
    # 4 5x5x2 images (stored in a 2x5x5x4 tensor)
    ddata = DummyDataset(axes=('c', 0, 1, 'b'))
    topo = ddata.get_topological_view()

    # ref_win[b]: a set of hashes, computed from all possible 3x3 windows of
    #             topo[..., b].
    ref_win = [set() for _ in xrange(4)]
    #ref_win = [[] for _ in xrange(4)]
    for b in xrange(topo.shape[-1]):
        # get all possible 3x3 windows within the 5x5 images.
        for i in xrange(3):
            for j in xrange(3):
                window = topo[:, i:i + 3, j:j + 3, b]
                assert_equal((3, 3), window.shape[1:])
                # Add a SHA1 digest of the window to set ref_win[b]
                ref_win[b].add(_hash_array(window))
                if flip:
                    # Also add the hash of the window with axis 1 flipped
                    ref_win[b].add(_hash_array(window[:, :, ::-1]))
    actual_win = [set() for _ in xrange(4)]

    if use_old_c01b:
        wf_cls = WindowAndFlipC01B
    else:
        wf_cls = WindowAndFlip

    # no zero-padding.
    wf = wf_cls(window_shape=(3, 3), randomize=[ddata], flip=flip)
    wf.setup(None, ddata, None)  # ddata argument is ignored

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
            assert_contains(ref_win[b], hashed)
            actual_win[b].add(hashed)


def check_window_flip_coverage_B01C(flip):
    ddata = DummyDataset(axes=('b', 0, 1, 'c'))
    topo = ddata.get_topological_view()
    ref_win = [set() for _ in xrange(4)]
    for b in xrange(topo.shape[0]):
        for i in xrange(3):
            for j in xrange(3):
                window = topo[b, i:i + 3, j:j + 3, :]
                assert_equal((3, 3), window.shape[:-1])
                ref_win[b].add(_hash_array(window))
                if flip:
                    ref_win[b].add(_hash_array(window[:, ::-1, :]))
    actual_win = [set() for _ in xrange(4)]
    wf = WindowAndFlip(window_shape=(3, 3), randomize=[ddata], flip=flip)
    wf.setup(None, ddata, None)
    curr_topo = ddata.get_topological_view()
    assert_equal((4, 3, 3, 2), curr_topo.shape)
    for b in xrange(topo.shape[0]):
        hashed = _hash_array(curr_topo[b, ...])
        assert_contains(ref_win[b], hashed)
        actual_win[b].add(hashed)
    while not all(len(a) == len(b) for a, b in zip(ref_win, actual_win)):
        prev_topo = curr_topo.copy()
        wf.on_monitor(None, ddata, None)
        curr_topo = ddata.get_topological_view()
        assert_(not (prev_topo == curr_topo).all())
        for b in xrange(topo.shape[0]):
            hashed = _hash_array(curr_topo[b, ...])
            assert_contains(ref_win[b], hashed)
            actual_win[b].add(hashed)


def test_padding():
    # Old interface WindowAndFlipC01B
    yield check_padding, ('c', 0, 1, 'b'), True
    # New interface WindowAndFlip
    yield check_padding, ('c', 0, 1, 'b')
    yield check_padding, ('b', 0, 1, 'c')


def check_padding(axes, use_old_c01b=False):

    padding = 3
    ddata = DummyDataset()
    topo = ddata.get_topological_view()

    if use_old_c01b:
        wf_cls = WindowAndFlipC01B
    else:
        wf_cls = WindowAndFlip

    wf = wf_cls(window_shape=(5, 5), randomize=[ddata],
                           pad_randomized=padding)
    wf.setup(None, None, None)
    new_topo = ddata.get_topological_view()
    assert_equal(topo.shape, new_topo.shape)
    saw_padding = dict([((direction, amount), False) for direction, amount
                        in itertools.product(['l', 'b', 'r', 't'],
                                             xrange(padding))])
    iters = 0
    while not all(saw_padding.values()) and iters < 50:
        for image in new_topo.swapaxes(0, 3):
            for i in xrange(padding):
                if (image[:i] == 0).all():
                    saw_padding['t', i] = True
                if (image[-i:] == 0).all():
                    saw_padding['b', i] = True
                if (image[:, -i:] == 0).all():
                    saw_padding['r', i] = True
                if (image[:, :i] == 0).all():
                    saw_padding['l', i] = True
        wf.on_monitor(None, None, None)
        new_topo = ddata.get_topological_view()
        iters += 1


def test_WindowAndFlipC01B_axes_guard():
    ddata = DummyDataset(axes=('b', 0, 1, 'c'))
    raised_error = False
    try:
        wf = WindowAndFlipC01B(window_shape=(3, 3), randomize=[ddata])
    except ValueError:
        raised_error = True
    assert_equal(raised_error, True)
