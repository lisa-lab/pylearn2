__author__ = "Ian Goodfellow"

from matplotlib import pyplot
import sys
from theano.compat.six.moves import xrange
pyplot.hold(True)

from pylearn2.utils import serial

model_paths = sys.argv[1:]

smoothing = 1
try:
    smoothing = int(model_paths[0])
    model_paths = model_paths[1:]
except Exception:
    pass

count = 0
style = '-'
for model_path in model_paths:
    model = serial.load(model_path)
    smoothed_reward_record = []
    count += 1
    if count > 7:
        style = '+'
    for i in xrange(smoothing - 1, len(model.reward_record)):
        smoothed_reward_record.append(sum(model.reward_record[i - smoothing + 1:i + 1]) / float(smoothing))
    pyplot.plot(smoothed_reward_record, style, label=model_path)
pyplot.legend()
pyplot.show()


