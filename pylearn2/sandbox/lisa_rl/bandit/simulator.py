__author__ = "Ian Goodfellow"

import logging
import numpy as np

from pylearn2.utils import serial
from pylearn2.utils import contains_nan, contains_inf


logger = logging.getLogger(__name__)


class Simulator(object):
    """
    .. todo::

        WRITEME : parameter list
    """
    def __init__(self, agent, environment, algorithm, save_path):
        self.__dict__.update(locals())
        del self.self

    def main_loop(self):
        self.algorithm.setup(agent=self.agent, environment=self.environment)
        i = 0
        for param in self.agent.get_params():
            assert not contains_nan(param.get_value()), (i, param.name)
            assert not contains_inf(param.get_value()), (i, param.name)
        while True:
            rval = self.algorithm.train()
            assert rval is None
            i += 1
            for param in self.agent.get_params():
                assert not contains_nan(param.get_value()), (i, param.name)
                assert not contains_inf(param.get_value()), (i, param.name)
            if i % 1000 == 0:
                serial.save(self.save_path, self.agent)
                logger.info('saved!')
