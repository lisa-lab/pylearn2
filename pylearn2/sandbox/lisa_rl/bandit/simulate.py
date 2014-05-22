__author__ = "Ian Goodfellow"

from pylearn2.config import yaml_parse
import sys

_, path = sys.argv

simulator = yaml_parse.load_path(path)

simulator.main_loop()
