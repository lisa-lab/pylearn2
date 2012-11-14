# Local imports
import pylearn2.config.yaml_parse

import jobman
from jobman.tools import expand, flatten


class ydict(dict):
    '''
    YAML-friendly subclass of dictionary.

    The special key "__builder__" is interpreted as the name of an object
    constructor.

    For instance, building a ydict from the following dictionary:

        {
            '__builder__': 'pylearn2.training_algorithms.sgd.EpochCounter',
            'max_epochs': 2
        }

    Will be displayed like:

        !obj:pylearn2.training_algorithms.sgd.EpochCounter {'max_epochs': 2}
    '''
    def __str__(self):
        args_dict = dict(self)
        builder = args_dict.pop('__builder__', '')
        ret_list = []
        if builder:
            ret_list.append('!obj:%s {' % builder)
        else:
            ret_list.append('{')

        for key, val in args_dict.iteritems():
            # This will call str() on keys and values, not repr(), so unicode
            # objects will have the form 'blah', not "u'blah'".
            ret_list.append('%s: %s,' % (key, val))

        ret_list.append('}')
        return '\n'.join(ret_list)


def train_experiment(state, channel):
    """
    Train a model specified in state, and extract required results.

    This function builds a YAML string from ``state.yaml_template``, taking
    the values of hyper-parameters from ``state.hyper_parameters``, creates
    the corresponding object and trains it (like train.py), then run the
    function in ``state.extract_results`` on it, and store the returned values
    into ``state.results``.

    To know how to use this function, you can check the example in tester.py
    (in the same directory).
    """
    yaml_template = state.yaml_template

    # Convert nested DD into nested ydict.
    hyper_parameters = expand(flatten(state.hyper_parameters), dict_type=ydict)

    # This will be the complete yaml string that should be executed
    final_yaml_str = yaml_template % hyper_parameters

    # Instantiate an object from YAML string
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)

    try:
        iter(train_obj)
        iterable = True
    except TypeError:
        iterable = False
    if iterable:
        raise NotImplementedError(
                ('Current implementation does not support running multiple '
                 'models in one yaml string.  Please change the yaml template '
                 'and parameters to contain only one single model.'))
    else:
        # print "Executing the model."
        # (GD) HACK HACK
        train_obj.model.jobman_channel = channel
        train_obj.model.jobman_state = state
        train_obj.main_loop()
        return channel.COMPLETE
