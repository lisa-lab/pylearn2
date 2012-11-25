import logging
import sys

# By default, we add a StreamHandler here so that messages logged by
# pylearn2 (at log level INFO or above) get printed to sys.stdout.
# Scripts/users can override this by deleting the handler from the
# list of handlers on the pylearn2 logger object, e.g.
#
#    del logging.getLogger('pylearn2').handlers[0]
#
# TODO: make this configurable.
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))

# Make INFO the default log level.
_logger.setLevel(logging.INFO)

# This can always be retrieved by getLogger('pylearn2') so just remove
# the object reference for cleanliness.
del _logger, sys, logging
