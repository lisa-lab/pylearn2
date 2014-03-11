#/bin/sh

set -x -e

if [ "x$TEST_DOC" = "xYES" ]; then
    PYTHONPATH=${PYTHONPATH}:`pwd`

    ls `pwd`
    # does PYTHONPATH work?
    echo "CD experiment"
    mkdir a
    cd a
    mkdir b
    cd b
    python -c "import sys; print sys.path"
    python "import pylearn2"
    python "import pylearn2.config"
    python "import pylearn2.config.yaml_parse"
    python -c "import pylearn2.config.yaml_parse" || exit 1
    cd ../..
    excho "CD experiment concluded"

    python ./doc/scripts/docgen.py --test || exit 1
else
    # We can't build the test dataset as the original is not
    # present. We can't download the original as it is too big to
    # download each time. If present run: python make_dataset.py
    (cd pylearn2/scripts/tutorials/grbm_smd && wget http://www.iro.umontreal.ca/~lisa/datasets/cifar10_preprocessed_train.pkl)
    nosetests --help
    THEANO_FLAGS="$FLAGS",warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise TRAVIS=1 theano-nose -v $PART || exit 1
fi
