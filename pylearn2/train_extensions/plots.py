"""
Plot monitoring extensions while training.
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"


import logging
import os
import os.path
import stat
import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension

from theano.compat.six.moves import xrange

from pylearn2.utils import as_floatX, wraps

if os.getenv('DISPLAY') is None:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings

log = logging.getLogger(__name__)


def make_readable(fn):
    """
    Make a file readable by all.
    Practical when the plot is in your public_html.


    Parameters
    ----------
    fn : str
        Filename you wish to make public readable.
    """

    st = os.stat(fn)

    # Create the desired permission
    st_mode = st.st_mode
    read_all = stat.S_IRUSR
    read_all |= stat.S_IRGRP
    read_all |= stat.S_IROTH

    # Set the permission
    os.chmod(fn, st_mode | read_all)


def get_best_layout(n_plots):
    """
    Find the best basic layout for a given number of plots.
    Minimize the perimeter with a minimum area (``n_plots``) for
    an integer rectangle.


    Parameters
    ----------
    n_plots : int
        The number of plots to display

    Returns
    -------
    n_rows : int
        Number of rows in the layout
    n_cols :
        Number of columns in the layout
    """

    assert n_plots > 0

    # Initialize the layout
    n_rows = 1
    n_cols = np.ceil(n_plots*1./n_rows)
    n_cols = int(n_cols)
    half_perimeter = n_cols + 1

    # Limit the range of possible layouts
    max_row = np.sqrt(n_plots)
    max_row = np.round(max_row)
    max_row = int(max_row)

    for l in xrange(1, max_row + 1):
        width = np.ceil(n_plots*1./l)
        width = int(width)
        if half_perimeter >= (width + l):
            n_rows = l
            n_cols = np.ceil(n_plots*1./n_rows)
            n_cols = int(n_cols)
            half_perimeter = n_rows + n_cols

    return n_rows, n_cols


def create_colors(n_colors):
    """
    Create an array of n_colors


    Parameters
    ----------
    n_colors : int
        The number of colors to create

    Returns
    -------
    colors_rgb : np.array
        An array of shape (n_colors, 3) in RGB format
    """
    # Create the list of color hue
    colors_hue = np.arange(n_colors)
    colors_hue = as_floatX(colors_hue)
    colors_hue *= 1./n_colors

    # Set the color in HSV format
    colors_hsv = np.ones((n_colors, 3))
    colors_hsv[:, 2] *= .75
    colors_hsv[:, 0] = colors_hue

    # Put in a matplotlib-friendly format
    colors_hsv = colors_hsv.reshape((1, )+colors_hsv.shape)
    # Convert to RGB
    colors_rgb = matplotlib.colors.hsv_to_rgb(colors_hsv)
    colors_rgb = colors_rgb[0]

    return colors_rgb


class Plotter(object):
    """
    Base class for plotting.

    Parameters
    ----------
    freq : int, optional
        The number of epochs before producing plot.
        Default is None (set by the PlotManager).

    """
    def __init__(self, freq=None):
        self.filenames = []
        self.freq = freq

    def setup(self, model, dataset, algorithm):
        """
        Setup the plotters.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model trained
        dataset : pylearn2.datasets.Dataset
            The dataset on which the model is trained
        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The algorithm the model is trained with

        """
        raise NotImplementedError(str(type(self))+" does not implement setup.")

    def plot(self):
        """
        The method that draw and save the desired figure, which depend
        on the object and its attribute. This method is called by the
        PlotManager object as frequently as the `freq` attribute defines it.

        """
        raise NotImplementedError(str(type(self))+" does not implement plot.")

    def set_permissions(self, public):
        """
        Make the produced files readable by everyone.

        Parameters
        ----------
        public : bool
            If public is True, then the associated files are
            readable by everyone.

        """
        if public:
            for filename in self.filenames:
                make_readable(filename)


class Plots(Plotter):
    """
    Plot different monitors.

    Parameters
    ----------
    channel_names : list of str
        List of monitor channels to plot
    save_path : str
        Filename of the plot file
    share : float, optional
        The percentage of epochs shown. Default is .8 (80%)
    per_second : bool, optional
        Set if the x-axis is in seconds, in epochs otherwise.
        Default is False.
    kwargs : dict
        Passed on to the superclass.
    """
    def __init__(self, channel_names,
                 save_path, share=.8,
                 per_second=False,
                 ** kwargs):
        super(Plots, self).__init__(** kwargs)

        if not save_path.endswith('.png'):
            save_path += '.png'

        self.save_path = save_path
        self.filenames = [self.save_path]

        self.channel_names = channel_names
        self.n_colors = len(self.channel_names)
        self.colors_rgb = create_colors(self.n_colors)

        self.share = share
        self.per_second = per_second

    @wraps(Plotter.setup)
    def setup(self, model, dataset, algorithm):
        self.model = model

    @wraps(Plotter.plot)
    def plot(self):
        monitor = self.model.monitor
        channels = monitor.channels
        channel_names = self.channel_names

        # Accumulate the plots
        plots = np.array(channels[channel_names[0]].val_record)
        plots = plots.reshape((1, plots.shape[0]))
        plots = plots.repeat(self.n_colors, axis=0)
        for i, channel_name in enumerate(channel_names[1:]):
            plots[i+1] = np.array(channels[channel_name].val_record)

        # Keep the relevant part
        n_min = plots.shape[1]
        n_min -= int(np.ceil(plots.shape[1] * self.share))
        plots = plots[:, n_min:]

        # Get the x axis
        x = np.arange(plots.shape[1])
        x += n_min

        # Put in seconds if needed
        if self.per_second:
            seconds = channels['training_seconds_this_epoch'].val_record
            seconds = np.array(seconds)
            seconds = seconds.cumsum()
            x = seconds[x]

        # Plot the quantities
        plt.figure()
        for i in xrange(self.n_colors):
            plt.plot(x, plots[i], color=self.colors_rgb[i],
                     alpha=.5)

        plt.legend(self.channel_names)
        plt.xlim(x[0], x[-1])
        plt.ylim(plots.min(), plots.max())
        plt.axis('on')
        plt.savefig(self.save_path)
        plt.close()


class PlotManager(TrainExtension):
    """
    Class to manage the Plotter classes.

    Parameters
    ----------
    plots : list of pylearn2.train_extensions.Plotter
        List of plots to make during training
    freq : int
        The default number of epochs before producing plot.
    public : bool
        Whether the files are made public or not. Default is true.
    html_path : str
        The path where the HTML page is saved. The associated files should be
        in the same folder. Default is None, then there is no HTML page.
    """

    def __init__(self, plots, freq, public=True, html_path=None):

        self.plots = plots
        self.freq = freq

        # Set a default freq
        for plot in self.plots:
            if plot.freq is None:
                plot.freq = self.freq

        self.public = public
        self.html_path = html_path

        self.filenames = []

        self.count = 0

    @wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):

        for plot in self.plots:
            plot.setup(model, dataset, algorithm)
            for filename in plot.filenames:
                warn = ("/home/www-etud/" in filename)
                warn |= (os.environ['HOME'] in filename)
                warn &= ('umontreal' in os.environ['HOSTNAME'])
                if warn:
                    warnings.warn('YOU MIGHT RUIN THE NFS'
                                  'BY SAVING IN THIS PATH !')
                self.filenames.append(filename)

        if self.html_path is not None:
            header = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                      '<html xmlns="http://www.w3.org/1999/xhtml"'
                      'xml:lang="en">\n'
                      '\t<body>\n')
            footer = ('\t</body>\n'
                      '</html>')
            body = ''

            for filename in self.filenames:
                basename = os.path.basename(filename)
                body += '<img src = "' + basename + '"><br/>\n'

            with open(self.html_path, 'w') as f:
                f.write(header + body + footer)
                f.close()

            if self.public:
                make_readable(self.html_path)

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):

        self.count += 1
        for plot in self.plots:
            if self.count % plot.freq == 0:
                try:
                    plot.plot()
                    plot.set_permissions(self.public)
                except Exception as e:
                    warnings.warn(str(plot) + ' has failed.\n'
                                  + str(e))
