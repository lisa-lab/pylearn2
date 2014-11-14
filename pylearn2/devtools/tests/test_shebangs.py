from __future__ import print_function

__author__ = "Ian Goodfellow"

from pylearn2.devtools.list_files import list_files

def test_shebangs():
    # Make sure all scripts that use shebangs use /usr/bin/env
    # (instead of the non-standard /bin/env or hardcoding the path to
    # the interpreter). This test allows any shebang lines that start
    # with /usr/bin/env. Examples:
    #   "#!/usr/bin/env python"
    #   "#! /usr/bin/env python"
    #   "#!/usr/bin/env ipython"
    #   "#!/usr/bin/env ipython --pylab --"
    # etc.
    files = list_files('.py')
    for f in files:
        fd = open(f, 'r')
        l = fd.readline()
        fd.close()
        if l.startswith("#!"):
            if not l[2:].strip().startswith("/usr/bin/env"):
                print(l)
                print(f)
                raise AssertionError("Bad shebang")
