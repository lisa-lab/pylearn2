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


def total_seconds(delta):
    """
    Extract the total number of seconds from a timedelta object
    in a way that is compatible with Python <= 2.6.

    Parameters
    ----------
    delta : object
        A `datetime.timedelta` object.

    Returns
    -------
    total : float
        The time quantity represented by `delta` in seconds,
        with a fractional portion.
    """
    if hasattr(delta, 'total_seconds'):
        return delta.total_seconds()
    else:
        return (delta.microseconds +
                (delta.seconds + delta.days * 24 * 3600) * 10 ** 6
                ) / float(10 ** 6)


@contextmanager
def log_timing(logger, task, level=logging.INFO, final_msg=None,
               callbacks=None):
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
    callbacks: list, optional
        A list of callbacks taking as argument an
        integer representing the total number of seconds.
    """
    start = datetime.datetime.now()
    if task is not None:
        logger.log(level, str(task) + '...')
    yield
    end = datetime.datetime.now()
    delta = end - start
    total = total_seconds(delta)
    if total < 60:
        delta_str = '%f seconds' % total
    else:
        delta_str = str(delta)
    if final_msg is None:
        logger.log(level, str(task) + ' done. Time elapsed: %s' % delta_str)
    else:
        logger.log(level, ' '.join((final_msg, delta_str)))
    if callbacks is not None:
        for callback in callbacks:
            callback(total)
