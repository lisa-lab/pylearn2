#!/bin/bash
# This script downloads CIFAR-10 dataset to $PYLEARN2_DATA_PATH/cifar10
#set -e
[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1
CIFAR10_DIR=$PYLEARN2_DATA_PATH/cifar10

which wget > /dev/null
WGET=$?
which curl > /dev/null
CURL=$?

if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget --no-verbose -O -"
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl --silent -o -"
else
    echo "You need wget or curl installed to download"
    exit 1
fi

[ -d $CIFAR10_DIR ] && echo "$CIFAR10_DIR already exists." && exit 1
mkdir -p $CIFAR10_DIR

echo "Downloading and unzipping CIFAR-10 dataset into $CIFAR10_DIR..."
pushd $CIFAR10_DIR > /dev/null
$DL_CMD http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz | tar xvzf -
popd > /dev/null
