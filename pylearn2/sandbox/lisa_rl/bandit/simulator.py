__author__ = "Ian Goodfellow"

from pylearn2.utils import serial

class Simulator(object):

    def __init__(self, agent, environment, algorithm, save_path):
        self.__dict__.update(locals())
        del self.self

    def main_loop(self):
        self.algorithm.setup(agent=self.agent, environment=self.environment)
        i = 0
        while True:
            rval = self.algorithm.train()
            assert rval is None
            i += 1
            if i % 1000 == 0:
                serial.save(self.save_path, self.agent)
                print 'saved!'


