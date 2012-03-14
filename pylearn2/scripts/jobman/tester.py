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
    This is a user specific functioni that is used by jobmman to extract results such that the returned dictionary will
    be saved in state.results
    """   
    import numpy
    channels = train_obj.model.monitor.channels['sgd_cost(UnsupervisedExhaustiveSGD[X])']    
    #This function returns the reconstruction_error and the bach numbers for CAE
    return dict(reconstruction_error= channels.val_record , batch_num= channels.batch_record)


if __name__ == '__main__':
    TABLE_NAME = 'test_jobman_pylearn2'
    db = api0.open_db('postgres://almouslh@gershwin.iro.umontreal.ca/almouslh_db?table='+TABLE_NAME)

    state = DD()

    state.yaml_template = '''
        !obj:pylearn2.scripts.train.Train {
        "dataset": !obj:pylearn2.datasets.npy_npz.NpyDataset &dataset {
            "file" : "%(file)s" # Should be an N x 300 matrix on disk.
        },
        "model": !obj:pylearn2.autoencoder.ContractiveAutoencoder {
            "nvis" : %(nvis)d,
            "nhid" : %(nhid)d,
            "irange" : 0.05,
            "act_enc": "sigmoid", #for some reason only sigmoid function works
            "act_dec": "sigmoid",
        },
        "algorithm": !obj:pylearn2.training_algorithms.sgd.UnsupervisedExhaustiveSGD {
            "learning_rate" : %(learning_rate)f,
            "batch_size" : %(batch_size)d,
            "monitoring_batches" : 5,
            "monitoring_dataset" : *dataset,
            "cost" : [!obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {},
            !obj:pylearn2.costs.autoencoder.ScaleBy {
              cost: !obj:pylearn2.costs.autoencoder.ModelMethodPenalty {
                method_name: contraction_penalty
                    },
                    coefficient: %(coefficient)f } ],
            "termination_criterion" : %(term_crit_builder)s %(term_crit_args)s,
        }
    }
    '''

    state.hyper_parameters = {
            "file": "/u/almouslh/Documents/UTLCChallenge/Version23Feb/ift6266h12/ift6266h12/experiments/sylvester/cae/layers/npy/sylvester_train_pca8.npy",
            "nvis": 8,
            "nhid": 6,
            "learning_rate": 0.1,
            "batch_size": 10,
            "coefficient": 0.5,
            "term_crit_builder": "!obj:pylearn2.training_algorithms.sgd.EpochCounter",
            "term_crit_args": {
                "max_epochs": 2
                }
            }

    state.extract_results = "pylearn2.scripts.jobman.tester.result_extractor"

    sql.insert_job(experiment.train_experiment, flatten(state), db, force_dup=True)
