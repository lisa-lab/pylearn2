from pylearn2.devtools.run_pyflakes import run_pyflakes

def test_via_pyflakes():
    d = run_pyflakes(True)
    if len(d.keys()) != 0:
        print 'Errors detected by pyflakes'
        for key in d.keys():
            print key+':'
            for l in d[key].split('\n'):
                print '\t',l

        raise AssertionError("You have errors detected by pyflakes")
