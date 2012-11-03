"""Utilities related to timing various segments of code."""

__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

from contextlib import contextmanager
import logging
import datetime


@contextmanager
def log_timing(logger, task, level=logging.INFO, final_msg=None):
    """
    Context manager that logs the start/end of an operation,
    and timing information, to a given logger.

    Parameters
    ----------
    logger : object
        A Python standard library logger object, or an object
        that supports the `logger.log(level, message, ...)`
        API it defines.

    task : str
        A string indicating the operation being performed.
        A '...' will be appended to the initial logged message.
        If `None`, no initial message will be printed.

    level : int, optional
        The log level to use. Default `logging.INFO`.

    final_msg : str, optional
        Display this before the reported time instead of
        '<task> done. Time elapsed:'. A space will be
        added between this message and the reported
        time.
    """
    start = datetime.datetime.now()
    if task is not None:
        logger.log(level, str(task) + '...')
    yield
    end = datetime.datetime.now()
    delta = end - start
    if delta.total_seconds() < 60:
        delta_str = '%f seconds' % delta.total_seconds()
    else:
        delta_str = str(delta)
    if final_msg is None:
        logger.log(level, str(task) + ' done. Time elapsed: %s' % delta_str)
    else:
        logger.log(level, ' '.join((final_msg, delta_str)))
