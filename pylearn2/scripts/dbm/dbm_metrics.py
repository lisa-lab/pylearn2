#!/usr/bin/env python
import argparse
from pylearn2.utils import serial


def compute_ais(model):
    pass

if __name__ == '__main__':
    # Possible metrics
    metrics = {'ais': compute_ais}

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="the desired metric",
                        choices=metrics.keys())
    parser.add_argument("model_path", help="path to the pickled DBM model")
    args = parser.parse_args()

    metric = metrics[args.metric]
    model = serial.load(args.model_path)

    metric(model)
