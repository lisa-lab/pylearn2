__author__ = "Ian Goodfellow"

from matplotlib import pyplot
import sys

from pylearn2.utils import serial

_, model_path = sys.argv

model = serial.load(model_path)

pyplot.plot(model.reward_record)
pyplot.show()
