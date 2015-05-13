"""
Training extension for allowing querying of monitoring values while an
experiment executes.
"""
__authors__ = "Dustin Webb, Adam Stone"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Dustin Webb,  Adam Stone"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import copy
import logging
log = logging.getLogger('LiveMonitor')

try:
    import zmq
    zmq_available = True
except Exception:
    zmq_available = False

try:
    from PySide import QtCore, QtGui

    import sys
    import matplotlib
    import numpy as np
    matplotlib.use('Qt4Agg')
    matplotlib.rcParams['backend.qt4'] = 'PySide'

    from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg as FigureCanvas)
    from matplotlib.figure import Figure

    qt_available = True
except Exception:
    qt_available = False

try:
    import matplotlib.pyplot as plt
    pyplot_available = True
except Exception:
    pyplot_available = False

from functools import wraps
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension


class LiveMonitorMsg(object):
    """
    Base class that defines the required interface for all Live Monitor
    messages.
    """
    response_set = False

    def get_response(self):
        """
        Method that instantiates a response message for a given request
        message. It is not necessary to implement this function on response
        messages.
        """
        raise NotImplementedError('get_response is not implemented.')


class ChannelListResponse(LiveMonitorMsg):
    """
    A message containing the list of channels being monitored.
    """
    pass


class ChannelListRequest(LiveMonitorMsg):
    """
    A message indicating a request for a list of channels being monitored.
    """
    @wraps(LiveMonitorMsg.get_response)
    def get_response(self):
        return ChannelListResponse()


class ChannelsResponse(LiveMonitorMsg):
    """
    A message containing monitoring data related to the channels specified.
    Data can be requested for all epochs or select epochs.

    Parameters
    ----------
    channel_list : list
        A list of the channels for which data has been requested.

    start : int
        The starting epoch for which data should be returned.

    end : int
        The epoch after which data should be returned.

    step : int
        The number of epochs to be skipped between data points.
    """
    def __init__(self, channel_list, start, end, step=1):
        assert(
            isinstance(channel_list, list)
            and len(channel_list) > 0
        )
        self.channel_list = channel_list

        assert(start >= 0)
        self.start = start

        self.end = end

        assert(step > 0)
        self.step = step


class ChannelsRequest(LiveMonitorMsg):
    """
    A message for requesting data related to the channels specified.

    Parameters
    ----------
    channel_list : list
        A list of the channels for which data has been requested.

    start : int
        The starting epoch for which data should be returned.

    end : int
        The epoch after which data should be returned.

    step : int
        The number of epochs to be skipped between data points.
    """
    def __init__(self, channel_list, start=0, end=-1, step=1):
        assert(
            isinstance(channel_list, list)
            and len(channel_list) > 0
        )
        self.channel_list = channel_list

        assert(start >= 0)
        self.start = start

        self.end = end

        assert(step > 0)
        self.step = step

    @wraps(LiveMonitorMsg.get_response)
    def get_response(self):
        return ChannelsResponse(
            self.channel_list,
            self.start,
            self.end,
            self.step
        )


class LiveMonitoring(TrainExtension):
    """
    A training extension for remotely monitoring and filtering the channels
    being monitored in real time. PyZMQ must be installed for this extension
    to work.

    Parameters
    ----------
    address : string
        The IP addresses of the interfaces on which the monitor should listen.

    req_port : int
        The port number to be used to service request.

    pub_port : int
        The port number to be used to publish updates.
    """
    def __init__(self, address='*', req_port=5555, pub_port=5556):
        if not zmq_available:
            raise ImportError('zeromq needs to be installed to '
                              'use this module.')

        self.address = 'tcp://%s' % address

        assert(req_port != pub_port)

        assert(req_port > 1024 and req_port < 65536)
        self.req_port = req_port

        assert(pub_port > 1024 and pub_port < 65536)
        self.pub_port = pub_port

        address_template = self.address + ':%d'
        self.context = zmq.Context()

        self.req_sock = None
        if self.req_port > 0:
            self.req_sock = self.context.socket(zmq.REP)
            self.req_sock.bind(address_template % self.req_port)

        self.pub_sock = None
        if self.pub_port > 0:
            self.pub_sock = self.context.socket(zmq.PUB)
            self.req_sock.bind(address_template % self.pub_port)

        # Tracks the number of times on_monitor has been called
        self.counter = 0

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        monitor = Monitor.get_monitor(model)
        try:
            rsqt_msg = self.req_sock.recv_pyobj(flags=zmq.NOBLOCK)

            # Determine what type of message was received
            rsp_msg = rsqt_msg.get_response()

            if isinstance(rsp_msg, ChannelListResponse):
                rsp_msg.data = list(monitor.channels.keys())

            if isinstance(rsp_msg, ChannelsResponse):
                channel_list = rsp_msg.channel_list
                if (
                        not isinstance(channel_list, list)
                        or len(channel_list) == 0
                ):
                    channel_list = []
                    result = TypeError(
                        'ChannelResponse requires a list of channels.'
                    )

                result = {}
                for channel_name in channel_list:
                    if channel_name in monitor.channels.keys():
                        chan = copy.deepcopy(
                            monitor.channels[channel_name]
                        )
                        end = rsp_msg.end
                        if end == -1:
                            end = len(chan.batch_record)
                        # TODO copying and truncating the records individually
                        # like this is brittle. Is there a more robust
                        # solution?
                        chan.batch_record = chan.batch_record[
                            rsp_msg.start:end:rsp_msg.step
                        ]
                        chan.epoch_record = chan.epoch_record[
                            rsp_msg.start:end:rsp_msg.step
                        ]
                        chan.example_record = chan.example_record[
                            rsp_msg.start:end:rsp_msg.step
                        ]
                        chan.time_record = chan.time_record[
                            rsp_msg.start:end:rsp_msg.step
                        ]
                        chan.val_record = chan.val_record[
                            rsp_msg.start:end:rsp_msg.step
                        ]
                        result[channel_name] = chan
                    else:
                        result[channel_name] = KeyError(
                            'Invalid channel: %s' % rsp_msg.channel
                        )
                rsp_msg.data = result

            self.req_sock.send_pyobj(rsp_msg)
        except zmq.Again:
            pass

        self.counter += 1


class LiveMonitor(object):
    """
    A utility class for requested data from a LiveMonitoring training
    extension.

    Parameters
    ----------
    address : string
        The IP address on which a LiveMonitoring process is listening.

    req_port : int
        The port number on which a LiveMonitoring process is listening.
    """
    def __init__(self, address='127.0.0.1', req_port=5555):
        """
        """
        if not zmq_available:
            raise ImportError('zeromq needs to be installed to '
                              'use this module.')

        self.address = 'tcp://%s' % address

        assert(req_port > 0)
        self.req_port = req_port

        self.context = zmq.Context()

        self.req_sock = self.context.socket(zmq.REQ)
        self.req_sock.connect(self.address + ':' + str(self.req_port))

        self.channels = {}

    def list_channels(self):
        """
        Returns a list of the channels being monitored.
        """
        self.req_sock.send_pyobj(ChannelListRequest())
        return self.req_sock.recv_pyobj()

    def update_channels(self, channel_list, start=-1, end=-1, step=1):
        """
        Retrieves data for a specified set of channels and combines that data
        with any previously retrived data.

        This assumes all the channels have the same number of values. It is
        unclear as to whether this is a reasonable assumption. If they do not
        have the same number of values then it may request to much or too
        little data leading to duplicated data or wholes in the data
        respectively. This could be made more robust by making a call to
        retrieve all the data for all of the channels.

        Parameters
        ----------
        channel_list : list
            A list of the channels for which data should be requested.

        start : int
            The starting epoch for which data should be requested.

        step : int
            The number of epochs to be skipped between data points.
        """
        assert((start == -1 and end == -1) or end > start)

        if start == -1:
            start = 0
            if len(self.channels.keys()) > 0:
                channel_name = list(self.channels.keys())[0]
                start = len(self.channels[channel_name].epoch_record)

        self.req_sock.send_pyobj(ChannelsRequest(
            channel_list, start=start, end=end, step=step
        ))

        rsp_msg = self.req_sock.recv_pyobj()

        if isinstance(rsp_msg.data, Exception):
            raise rsp_msg.data

        for channel in rsp_msg.data.keys():
            rsp_chan = rsp_msg.data[channel]

            if isinstance(rsp_chan, Exception):
                raise rsp_chan

            if channel not in self.channels.keys():
                self.channels[channel] = rsp_chan
            else:
                chan = self.channels[channel]
                chan.batch_record += rsp_chan.batch_record
                chan.epoch_record += rsp_chan.epoch_record
                chan.example_record += rsp_chan.example_record
                chan.time_record += rsp_chan.time_record
                chan.val_record += rsp_chan.val_record

    def follow_channels(self, channel_list, use_qt=False):
        """
        Tracks and plots a specified set of channels in real time.

        Parameters
        ----------
        channel_list : list
            A list of the channels for which data has been requested.
        use_qt : bool
            Use a PySide GUI for plotting, if available.
        """
        if use_qt:
            if not qt_available:
                log.warning(
                    'follow_channels called with use_qt=True, but PySide '
                    'is not available. Falling back on matplotlib ion().')
            else:
                # only create new qt app if running the first time in session
                if not hasattr(self, 'gui'):
                    self.gui = LiveMonitorGUI(self, channel_list)

                self.gui.channel_list = channel_list
                self.gui.start()

        elif not pyplot_available:
            raise ImportError('pyplot needs to be installed for '
                              'this functionality.')
        else:
            plt.clf()
            plt.ion()
            while True:
                self.update_channels(channel_list)
                plt.clf()
                for channel_name in self.channels:
                    plt.plot(
                        self.channels[channel_name].epoch_record,
                        self.channels[channel_name].val_record,
                        label=channel_name
                    )
                plt.legend()
                plt.ion()
                plt.draw()

if qt_available:
    class LiveMonitorGUI(QtGui.QMainWindow):
        def __init__(self, lm, channel_list):
            """
            PySide GUI implementation for live monitoring channels.

            Parameters
            ----------
            lm : LiveMonitor instance
                The LiveMonitor instance to which the GUI belongs.

            channel_list : list
                A list of the channels to display.
            """
            self.app = QtGui.QApplication(sys.argv)

            super(LiveMonitorGUI, self).__init__()
            self.lm = lm
            self.channel_list = channel_list
            self.updaterThread = UpdaterThread(lm, channel_list)
            self.updaterThread.updated.connect(self.refresh)
            self.initUI()

        def initUI(self):
            self.resize(300, 200)
            self.fig = Figure(figsize=(300, 200), dpi=72,
                              facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvas(self.fig)
            self.setCentralWidget(self.canvas)

        def refresh(self):
            self.ax.cla()  # clear previous plot
            for channel_name in self.channel_list:

                X = epoch_record = self.lm.channels[channel_name].epoch_record
                Y = val_record = self.lm.channels[channel_name].val_record

                indices = np.nonzero(np.diff(epoch_record))[0] + 1
                epoch_record_split = np.split(epoch_record, indices)
                val_record_split = np.split(val_record, indices)

                X = np.zeros(len(epoch_record))
                Y = np.zeros(len(epoch_record))

                for i, epoch in enumerate(epoch_record_split):

                    j = i*len(epoch_record_split[0])
                    X[j: j + len(epoch)] = (
                        1.*np.arange(len(epoch)) / len(epoch) + epoch[0])
                    Y[j: j + len(epoch)] = val_record_split[i]

                self.ax.plot(X, Y, label=channel_name)

            self.ax.legend()
            self.canvas.draw()
            self.updaterThread.start()

        def start(self):
            self.show()
            self.updaterThread.start()
            self.app.exec_()

    class UpdaterThread(QtCore.QThread):
        updated = QtCore.Signal()

        def __init__(self, lm, channel_list):
            super(UpdaterThread, self).__init__()
            self.lm = lm
            self.channel_list = channel_list

        def run(self):
            self.lm.update_channels(self.channel_list)  # blocking
            self.updated.emit()
