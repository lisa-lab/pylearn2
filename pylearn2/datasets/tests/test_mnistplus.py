"""module for testing datasets.mnistplus"""
from pylearn2.datasets.mnistplus import MNISTPlus
from pylearn2.testing.skip import skip_if_no_data
import string


digs = string.digits + string.lowercase

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x==0:
        return '0'
    else:
        sign = 1
    x *= sign
    digits = []
    while x:
        digits.append(digs[x % base])
        x /= base
    if sign < 0:
        digits.append('-')
    digits.reverse()
    return ''.join(digits)


def test_mnistplus():
    """
    Tests every combination of parameters for loading mnistplus.
    This test takes ~16.5 minutes on eos4.
    """
    skip_if_no_data()
    params = []
    for i in range(32):
        d = {}
        l = list(int(j) for j in list(int2base(i,2)))
        l = [0,]*(5-len(l)) + l
        d['azimuth'] = l[0]
        d['rotation'] = l[1]
        d['texture'] = l[2]
        d['center'] = l[3]
        d['contrast_normalize'] = l[4]
        params.append(d)
    for which_set in ['train','test','valid']:
        for p in params:
            for label_type in ['label','azimuth','rotation','texture_id']:
                _label = label_type
                if _label == 'texture_id': _label = 'texture'
                print which_set, p, label_type
                if label_type == 'label' or p[_label] == 1:
                    print 'loading...'
                    data = MNISTPlus(which_set=which_set,
                                     label_type=label_type,
                                     **p)
                    #assert data.X.min() >= 0.0
                    #assert data.X.max() <= 1.0
                    topo = data.get_batch_topo(1)
                    assert topo.ndim == 4
