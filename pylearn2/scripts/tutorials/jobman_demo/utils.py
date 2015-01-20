from __future__ import print_function

import numpy
from jobman import tools
from jobman.tools import DD


def log_uniform(low, high):
    """
    Generates a number that's uniformly distributed in the log-space between
    `low` and `high`

    Parameters
    ----------
    low : float
        Lower bound of the randomly generated number
    high : float
        Upper bound of the randomly generated number

    Returns
    -------
    rval : float
        Random number uniformly distributed in the log-space specified by `low`
        and `high`
    """
    log_low = numpy.log(low)
    log_high = numpy.log(high)

    log_rval = numpy.random.uniform(log_low, log_high)
    rval = float(numpy.exp(log_rval))

    return rval


def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    train_y_misclass = channels['y_misclass'].val_record[-1]
    train_y_nll = channels['y_nll'].val_record[-1]

    return DD(train_y_misclass=train_y_misclass,
              train_y_nll=train_y_nll)


def parse_results(cwd):
    optimal_dd = None
    optimal_measure = numpy.inf

    for tup in tools.find_conf_files(cwd):
        dd = tup[1]
        if 'results.train_y_misclass' in dd:
            if dd['results.train_y_misclass'] < optimal_measure:
                optimal_measure = dd['results.train_y_misclass']
                optimal_dd = dd

    print("Optimal results.train_y_misclass:", str(optimal_measure))
    for key, value in optimal_dd.items():
        if 'hyper_parameters' in key:
            print(key + ": " + str(value))
