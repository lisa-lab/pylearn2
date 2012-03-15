"""
This an example script inserting a pylearn2 yaml code into a jobman database.

The code below defines a yaml template string in state.yaml_template,
and the values of its hyper-parameters in state.hyper_parameters, and
run the code that is located in state.extract_results on this model
using jobman.

Actually, we add the job here and it can be launched later as usual
(please check how to start jobs using jobman from the jobman tutorial
website)

"""

from jobman.tools import DD, flatten
from jobman import api0, sql

from pylearn2.scripts.jobman import experiment


def result_extractor(train_obj):
    """
    This is a user specific function, that is used by jobman to extract results

    The returned dictionary will be saved in state.results
    """
    import numpy

    channels = train_obj.model.monitor.channels
    train_cost = channels['sgd_cost(ExhaustiveSGD[X])']
    best_epoch = numpy.argmin(train_cost.val_record)
    best_rec_error = train_cost.val_record[best_epoch]
    batch_num = train_cost.batch_record[best_epoch]
    return dict(
            best_epoch=best_epoch,
            train_rec_error=best_rec_error,
            batch_num=batch_num)


if __name__ == '__main__':
    db = api0.open_db('sqlite:///test.db?table=test_jobman_pylearn2')

    state = DD()

    state.yaml_template = '''
        !obj:pylearn2.scripts.train.Train {
        "dataset": !obj:pylearn2.datasets.npy_npz.NpyDataset &dataset {
            "file" : "%(file)s"
        },
        "model": !obj:pylearn2.autoencoder.ContractiveAutoencoder {
            "nvis" : %(nvis)d,
            "nhid" : %(nhid)d,
            "irange" : 0.05,
            "act_enc": "sigmoid", #for some reason only sigmoid function works
            "act_dec": "sigmoid",
        },
        "algorithm": !obj:pylearn2.training_algorithms.sgd.ExhaustiveSGD {
            "learning_rate" : %(learning_rate)f,
            "batch_size" : %(batch_size)d,
            "monitoring_batches" : 5,
            "monitoring_dataset" : *dataset,
            "cost" : [
                !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {},
                !obj:pylearn2.costs.autoencoder.ScaleBy {
                    cost: !obj:pylearn2.costs.autoencoder.ModelMethodPenalty {
                        method_name: contraction_penalty
                    },
                    coefficient: %(coefficient)f
                }
            ],
            "termination_criterion" : %(term_crit)s,
        }
    }
    '''

    state.hyper_parameters = {
            "file": "${PYLEARN2_DATA_PATH}/UTLC/pca/sylvester_train_x_pca32.npy",
            "nvis": 32,
            "nhid": 6,
            "learning_rate": 0.1,
            "batch_size": 10,
            "coefficient": 0.5,
            "term_crit": {
                "__builder__": "pylearn2.training_algorithms.sgd.EpochCounter",
                "max_epochs": 2
                }
            }

    state.extract_results = "pylearn2.scripts.jobman.tester.result_extractor"

    sql.insert_job(
            experiment.train_experiment,
            flatten(state),
            db,
            force_dup=True)
