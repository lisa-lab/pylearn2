
def result_extractor(train_obj):
    """
    This is a user specific functioni that is used by jobmman to extract results such that the returned dictionary will
    be saved in state.results
    """   
    import numpy
    channels = train_obj.model.monitor.channels['sgd_cost(UnsupervisedExhaustiveSGD[X])']    
    #This function returns the reconstruction_error and the bach numbers for CAE
    return dict(reconstruction_error= channels.val_record , batch_num= channels.batch_record)