import logging
from pylearn2.devtools.run_pyflakes import run_pyflakes
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

logger = logging.getLogger(__name__)


def test_via_pyflakes():
    d = run_pyflakes(True)
    if len(d.keys()) != 0:
        logger.info('Errors detected by pyflakes')
        for key in d.keys():
            logger.info(key + ':')
            for l in d[key].split('\n'):
                logger.info('\t %s', l)

        raise AssertionError("You have errors detected by pyflakes")
