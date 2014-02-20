def test_get_topological_view():
    norb = SmallNORB('train')

    topo_data = norb.get_topological_view(single_tensor=False)
    assert isinstance(topo_data, tuple)
    assert len(topo_data) == 2

