"""General training script"""
# Standard library imports
import sys

# Local imports
try:
    import framework
except ImportError:
    print >>sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

import framework.config

class Train:
    def __init__(self, tag, dataset, model, algorithm):
        if tag != 'train':
            raise ValueError('train.py expects "train" tag, received"'+tag+'" tag')

        self.dataset, self.model, self.algorithm  = [ framework.config.resolve(x) for x in [dataset, model, algorithm] ]

    def main_loop(self):
        self.algorithm.train(self.model, self.dataset)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("train.py takes exactly one argument")

    config_file_path = sys.argv[1]

    config_dict = framework.config.load(config_file_path)

    train_obj = framework.config.checked_call(Train,config_dict)

    train_obj.main_loop()

