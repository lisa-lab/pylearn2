"""Tests for logger utils methods."""

import logging
from pylearn2.utils.logger import newline

logger = logging.getLogger(__name__)


def test_newline():
    # Save current properties
    handlers = logger.handlers
    level = logger.getEffectiveLevel()

    newline(logger)

    # Ensure that the logger didn't change
    assert handlers == logger.handlers
    assert level == logger.getEffectiveLevel()
