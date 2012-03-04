from pylearn2.devtools.run_pyflakes import run_pyflakes

def test_via_pyflakes():
    d = run_pyflakes(True)
    assert len(d.keys()) == 0
