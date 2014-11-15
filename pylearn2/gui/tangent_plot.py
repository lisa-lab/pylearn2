"""
Code for plotting curves with tangent lines.
"""

__author__ = "Ian Goodfellow"

try:
    from matplotlib import pyplot
except Exception:
    pyplot = None
from theano.compat.six.moves import xrange


def tangent_plot(x, y, s):
    """
    Plots a curve with tangent lines.

    Parameters
    ----------
    x : list
        List of x coordinates.
        Assumed to be sorted into ascending order, so that the tangent
        lines occupy 80 percent of the horizontal space between each pair
        of points.
    y : list
        List of y coordinates
    s : list
        List of slopes
    """

    assert isinstance(x, list)
    assert isinstance(y, list)
    assert isinstance(s, list)
    n = len(x)
    assert len(y) == n
    assert len(s) == n

    if pyplot is None:
        raise RuntimeError("Could not import pyplot, can't run this code.")

    pyplot.plot(x, y, color='b')

    if n == 0:
        pyplot.show()
        return

    pyplot.hold(True)

    # Add dummy entries so that the for loop can use the same code on every
    # entry
    if n == 1:
        x = [x[0] - 1] + x[0] + [x[0] + 1.]
    else:
        x = [x[0] - (x[1] - x[0])] + x + [x[-2] + (x[-1] - x[-2])]

    y = [0.] + y + [0]
    s = [0.] + s + [0]

    for i in xrange(1, n + 1):
        ld = 0.4 * (x[i] - x[i - 1])
        lx = x[i] - ld
        ly = y[i] - ld * s[i]
        rd = 0.4 * (x[i + 1] - x[i])
        rx = x[i] + rd
        ry = y[i] + rd * s[i]
        pyplot.plot([lx, rx], [ly, ry], color='g')

    pyplot.show()

if __name__ == "__main__":
    # Demo by plotting a quadratic function
    import numpy as np
    x = np.arange(-5., 5., .1)
    y = 0.5 * (x ** 2)
    x = list(x)
    y = list(y)
    tangent_plot(x, y, x)
