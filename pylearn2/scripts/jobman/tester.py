"""
This is the place where we start running a pylearn2 yaml code with jobman
The code below defines a yaml string in state.yaml_template and it's params in state.hyper_parameters
and run the code that is located in state.extract_results on this model using jobman. Actually, we add the
job here and it can be launched later as usual (please check how to start jobs using jobman from the jobman tutorial website)
"""

from jobman.tools import DD, flatten
from jobman import api0, sql
import experiment as exp


TABLE_NAME='testJobmanPylearn'
db = api0.open_db('postgres://almouslh@gershwin.iro.umontreal.ca/almouslh_db?table='+TABLE_NAME)




yaml = '''
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
        "termination_criterion" : !obj:pylearn2.training_algorithms.sgd.EpochCounter {
            "max_epochs": 2,
        },
    }
}
'''
state= DD()

dicParam={"file":"/u/almouslh/Documents/UTLCChallenge/Version23Feb/ift6266h12/ift6266h12/experiments/sylvester/cae/layers/npy/sylvester_train_pca8.npy","nvis":8,"nhid":6,"learning_rate":0.1,"batch_size":10,"coefficient":0.5}
state.yaml_template = yaml
state.hyper_parameters = dicParam
state.extract_results = "pylearn2.scripts.jobman.extract_results.result_extractor"
sql.insert_job(exp.experimentModel, flatten(state), db)



