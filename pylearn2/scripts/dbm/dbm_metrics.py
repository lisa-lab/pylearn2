#!/usr/bin/env python
import argparse

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="the desired metric",
                        choices=["ais"])
    parser.add_argument("model_path", help="path to the pickled DBM model")
    args = parser.parse_args()

    metric = args.metric
    model_path = args.model_path
