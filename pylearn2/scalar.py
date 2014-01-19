""" Miscellaneous ops: rectifier """

from theano import scalar
from theano.tensor import elemwise


class ScalarRectifier(scalar.UnaryScalarOp):
    """
    .. todo::

        WRITEME
    """
    @staticmethod
    def st_impl(x):
        """
        .. todo::

            WRITEME
        """
        return x * (x > 0.0)

    def impl(self, x):
        """
        .. todo::

            WRITEME
        """
        return ScalarRectifier.st_impl(x)

    def grad(self, (x,), (gz,)):
        """
        .. todo::

            WRITEME
        """
        return [x > 0.0]

    def c_code(self, node, name, (x,), (z,), sub):
        """
        .. todo::

            WRITEME
        """
        if node.inputs[0].type == scalar.float32:
            return """%(z)s = %(x)s < 0.0f ? 0.0 : %(x)s;""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z)s = %(x)s < 0.0 ? 0.0 : %(x)s;""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        v = super(ScalarRectifier, self).c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v

scalar_rectifier = ScalarRectifier(scalar.upgrade_to_float,
                                   name='scalar_rectifier')
rectifier = elemwise.Elemwise(scalar_rectifier, name='rectifier')
