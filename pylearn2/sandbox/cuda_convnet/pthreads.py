from theano.configparser import AddConfigVar, StrParam

AddConfigVar('pthreads.inc_dir',
        "location of pthread.h",
        StrParam(""))

AddConfigVar('pthreads.lib_dir',
        "location of library implementing pthreads",
        StrParam(""))

AddConfigVar('pthreads.lib',
        'name of the library that implements pthreads (e.g. "pthreadVC2" if using pthreadVC2.dll/.lib from pthreads-win32)',
        StrParam(""))

