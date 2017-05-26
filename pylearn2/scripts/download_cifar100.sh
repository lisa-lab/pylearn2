#!/bin/bash
# This script downloads CIFAR-100 dataset to $PYLEARN2_DATA_PATH/cifar100
#set -e
[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1
CIFAR100_DIR=$PYLEARN2_DATA_PATH/cifar100

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

[ -d $CIFAR100_DIR ] && echo "$CIFAR100_DIR already exists." && exit 1
mkdir -p $CIFAR100_DIR

echo "Downloading and unzipping CIFAR-100 dataset into $CIFAR100_DIR..."
pushd $CIFAR100_DIR > /dev/null
$DL_CMD http://www.cs.utoronto.ca/~kriz/cifar-100-python.tar.gz | tar xvzf -
popd > /dev/null
