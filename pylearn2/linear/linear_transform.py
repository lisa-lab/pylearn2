
class LinearTransform(object):
    """
        A generic class describing a LinearTransform. Derived classes may implement linear
        transformation as a dense matrix multiply, a convolution, etc.

        Classes inheriting from this should also inherit from TheanoLinear's LinearTransform
        This class does not directly inherit from TheanoLinear's LinearTransform because
        most LinearTransform classes in pylearn2 will inherit from a TheanoLinear derived
        class and don't want to end up inheriting from TheanoLinear by two paths

        This class is basically just here as a placeholder to show you what extra methods you
        need to add to make a TheanoLinear LinearTransform work with pylearn2"""

    def get_params(self):
        """ Return a set of parameters that govern the linear transformation """

        raise NotImplementedError()

    def get_weights_topo(self):
        """ Return a batch of filters, formatted topologically.
            This only really makes sense if you are working with a topological space,
            such as for a convolution operator.
            If your transformation is defined on a VectorSpace then some other class
            like a ViewConverter will need to transform your vector into a topological
            space; you are not responsible for doing so here."""

        raise NotImplementedError()
