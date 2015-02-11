#!/bin/bash
# This script downloads SVHN Format 2 dataset to $PYLEARN2_DATA_PATH/SVHN/format2
#set -e
[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1
SVHN_DIR=$PYLEARN2_DATA_PATH/SVHN/format2

which wget > /dev/null
WGET=$?
which curl > /dev/null
CURL=$?

if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget "
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl "
else
    echo "You need wget or curl installed to download"
    exit 1
fi

[ -d $SVHN_DIR ] && echo "$SVHN_DIR already exists." && exit 1
mkdir -p $SVHN_DIR

echo "Downloading SVHN dataset into $SVHN_DIR..."
pushd $SVHN_DIR > /dev/null
$DL_CMD http://ufldl.stanford.edu/housenumbers/train_32x32.mat
$DL_CMD http://ufldl.stanford.edu/housenumbers/test_32x32.mat
$DL_CMD http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
popd > /dev/null
echo "Downloading Completed. Note: The dataset is for non-commercial use only"
