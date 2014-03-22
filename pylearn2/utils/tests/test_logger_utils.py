"""Tests for logger utils methods."""

import logging
from pylearn2.utils.logger import newline

logger = logging.getLogger(__name__)


def test_newline():
    """
    Test the state of a the logger passed to the newline function.
    The state has to be the same.
    """
    # Save current properties
    handlers = logger.handlers
    level = logger.getEffectiveLevel()

    newline(logger)

    # Ensure that the logger didn't change
    assert handlers == logger.handlers
    assert level == logger.getEffectiveLevel()
