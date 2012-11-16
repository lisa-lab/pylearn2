from theano.compile import Mode
import theano
import numpy as np

class NanGuardMode(Mode):
    def __init__(self, nan_is_error, inf_is_error):
        def nan_check(i, node, fn):
            inputs = fn.inputs
            for x in inputs:
                if nan_is_error:
                    assert not np.any(np.isnan(x))
                if inf_is_error:
                    assert not np.any(np.isinf(x))
            fn()
            outputs = fn.outputs
            for x in outputs:
                if nan_is_error:
                    assert not np.any(np.isnan(x))
                if inf_is_error:
                    assert not np.any(np.isinf(x))
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [nan_check])
        super(NanGuardMode, self).__init__(wrap_linker, optimizer='fast_run')
