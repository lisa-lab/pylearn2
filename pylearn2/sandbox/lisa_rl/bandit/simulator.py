__author__ = "Ian Goodfellow"

import numpy as np

from pylearn2.utils import serial

class Simulator(object):

    def __init__(self, agent, environment, algorithm, save_path):
        self.__dict__.update(locals())
        del self.self

    def main_loop(self):
        self.algorithm.setup(agent=self.agent, environment=self.environment)
        i = 0
        for param in self.agent.get_params():
            assert not np.any(np.isnan(param.get_value())), (i, param.name)
            assert not np.any(np.isinf(param.get_value())), (i, param.name)
        while True:
            rval = self.algorithm.train()
            assert rval is None
            i += 1
            for param in self.agent.get_params():
                assert not np.any(np.isnan(param.get_value())), (i, param.name)
                assert not np.any(np.isinf(param.get_value())), (i, param.name)
            if i % 1000 == 0:
                serial.save(self.save_path, self.agent)
                print 'saved!'


