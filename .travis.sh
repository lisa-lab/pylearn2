#/bin/sh


if [ "xTEST_DOC" == "xYES"]; then
    python ./doc/scripts/docgen.py --test
else
    (cd pylearn2/scripts/tutorial/grbm_smd && wget http://www.iro.umontreal.ca/~lisa/datasets/cifar10_preprocessed_train.pkl)
    nosetests --help
    THEANO_FLAGS=$FLAGS,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise TRAVIS=1 theano-nose -v
fi
    