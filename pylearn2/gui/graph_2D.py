"""
Classes for making simple 2D visualizations.
"""
import numpy as N
from theano.compat.six.moves import xrange
from theano import config


class Graph2D(object):
    """
    A class for plotting simple graphs in two dimensions.

    Parameters
    ----------
    shape : tuple
        The shape of the display of the graph in (rows, cols)
        format. Units are pixels
    xlim : tuple
        A tuple specifying (xmin, xmax). This determines what
        portion of the real numbers are displayed in the graph.
    ycenter : float
        The coordinate of the center pixel along the y axis.
    """
    def __init__(self, shape, xlim, ycenter):
        self.xmin = 0.
        self.xmax = 0.
        self.set_shape(shape)
        self.set_xlim(xlim)
        self.set_ycenter(ycenter)

        self.components = []

    def set_shape(self, shape):
        """
        Sets the shape of the display (in pixels)

        Parameters
        ----------
        shape : tuple
            The (rows, columns) of the display.
        """
        self.rows = shape[0]
        self.cols = shape[1]



    def set_xlim(self, xlim):
        """
        Sets the range of space that is plotted in the graph.

        Parameters
        ----------
        xlim : tuple
            The range (xmin, xmax)
        """
        # x coordinate of center of leftmost pixel
        self.xmin = xlim[0]
        # x coordinate of center of rightmost pixel
        self.xmax = xlim[1]
        self.delta_x = (self.xmax-self.xmin)/float(self.cols-1)

    def set_ycenter(self, ycenter):
        """
        Sets the y coordinate of the central pixel of the display.

        Parameters
        ----------
        ycenter : float
            The desired coordinate.
        """
        self.delta_y = self.delta_x
        self.ymin = ycenter - (self.rows / 2) * self.delta_y
        self.ymax = self.ymin + (self.rows -1) * self.delta_y

    def render(self):
        """
        Renders the graph.

        Returns
        -------
        output : ndarray
            An ndarray in (rows, cols, RGB) format.
        """
        rval = N.zeros((self.rows, self.cols, 3))

        for component in self.components:
            rval = component.render( prev_layer = rval, parent = self )
            assert rval is not None

        return rval

    def get_coords_for_col(self, i):
        """
        Returns the coordinates of every pixel in column i of the
        graph.

        Parameters
        ----------
        i : int
            Column index

        Returns
        -------
        coords : ndarray
            A vector containing the real-number coordinates of every
            pixel in column i of the graph.
        """
        X = N.zeros((self.rows,2),dtype=config.floatX)
        X[:,0] = self.xmin + float(i) * self.delta_x
        X[:,1] = self.ymin + N.cast[config.floatX](N.asarray(range(self.rows-1,-1,-1))) * self.delta_y


        return X

class HeatMap(object):
    """
    A class for plotting 2-D functions as heatmaps.

    Parameters
    ----------
    f : callable
        A callable that takes a design matrix of 2D coordinates and returns a
        vector containing the function value at those coordinates
    normalizer : callable, optional
        None or a callable that takes a 2D numpy array and returns a 2D numpy
        array
    render_mode : str
        * 'o' : opaque.
        * 'r' : render only to the (r)ed channel
    """
    def __init__(self, f, normalizer=None, render_mode = 'o'):
        self.f = f
        self.normalizer = normalizer
        self.render_mode = render_mode

    def render(self, prev_layer, parent):
        """
        Renders the heatmap.

        Parameters
        ----------
        prev_layer : numpy ndarray
            An image that will be copied into the new output.
            The new image will be rendered on top of the first
            one, i.e., `prev_layer` will be visible through the
            new heatmap if the new heatmap is not rendered in
            fully opaque mode.
        parent : Graph2D
            A Graph2D object that defines the coordinate system
            of the heatmap.

        Returns
        -------
        img : The rendered heatmap
        """
        my_img = prev_layer * 0.0

        for i in xrange(prev_layer.shape[1]):
            X = parent.get_coords_for_col(i)
            f = self.f(X)
            if len(f.shape) == 1:
                for j in xrange(3):
                    my_img[:,i,j] = f
            else:
                my_img[:,i,:] = f
            #end if
        #end for i

        if self.normalizer is not None:
            my_img = self.normalizer(my_img)
            assert my_img is not None

        if self.render_mode == 'r':
            my_img[:,:,1:] = prev_layer[:,:,1:]
        elif self.render_mode == 'o':
            pass
        else:
            raise NotImplementedError()

        return my_img
