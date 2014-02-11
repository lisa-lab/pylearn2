#!/bin/bash
# This script downloads CIFAR-10 dataset to $PYLEARN2_DATA_PATH/cifar10
set -e
[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1
CIFAR10_DIR=$PYLEARN2_DATA_PATH/cifar10

[ -d $CIFAR10_DIR ] && echo "$CIFAR10_DIR already exists." && exit 1
mkdir -p $CIFAR10_DIR

echo "Downloading and unzipping CIFAR-10 dataset into $CIFAR10_DIR..."
pushd $CIFAR10_DIR > /dev/null
wget --no-verbose -O - http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz | tar xvzf -
popd > /dev/null
