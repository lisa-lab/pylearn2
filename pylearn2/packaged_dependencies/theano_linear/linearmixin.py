
class LinearMixin(object):
    """
    This class can be mixed in to an Op that is linear in its first input.

    Supposing this Op takes two inputs: x, y.  It can be written as a
    linear operation in x: z = x W(y)

    """

    def transpose(zlike, *inputs_1_to_n):
        """
        This function returns (zlike) transpose(W(y))
        """
        raise NotImplementedError('override-me')

    def grads_1_to_n(inputs, gzlist):
        raise NotImplementedError('override-me')

    def grad(self, inputs, gzlist):
        if len(gzlist) > 1:
            raise NotImplementedError()
        g_input0 = self.transpose(gzlist[0], *inputs[1:])
        return [g_input0] + self.grads_1_to_n(inputs, gzlist)


