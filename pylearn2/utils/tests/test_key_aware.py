from pylearn2.utils.key_aware import KeyAwareDefaultDict


def test_key_aware_default_dict():
    a = KeyAwareDefaultDict(str)
    assert a[5] == '5'
    assert a[4] == '4'
    assert a[(3, 2)] == '(3, 2)'
    try:
        b = KeyAwareDefaultDict()
        b[5]
    except KeyError:
        return
    assert False
