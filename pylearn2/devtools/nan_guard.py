from theano.compile import Mode
import theano
import numpy as np
from pylearn2.models.dbm import flatten

class NanGuardMode(Mode):
    def __init__(self, nan_is_error, inf_is_error):

        def do_check_on(var, nd, f, is_input):
            error = False
            if nan_is_error:
                if np.any(np.isnan(var)):
                    print 'NaN detected'
                    error = True
            if inf_is_error:
                if np.any(np.isinf(var)):
                    print 'Inf detected'
                    error = True
            if np.abs(var).max() > 1e10:
                print 'Big value detected'
                error = True
            if error:
                if is_input:
                    print 'In an input'
                else:
                    print 'In an output'
                print 'Inputs: '
                for ivar, ival in zip(nd.inputs, f.inputs):
                    print 'var'
                    print ivar
                    print theano.printing.min_informative_str(ivar)
                    print 'val'
                    print ival
                print 'Node:'
                print nd
                assert False

        def nan_check(i, node, fn):
            inputs = fn.inputs
            # TODO: figure out why individual inputs are themselves lists sometimes
            for x in flatten(inputs):
                do_check_on(x, node, fn, True)
            fn()
            outputs = fn.outputs
            for i, x in enumerate(flatten(outputs)):
                do_check_on(x, node, fn, False)

        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [nan_check])
        super(NanGuardMode, self).__init__(wrap_linker, optimizer='fast_run')
