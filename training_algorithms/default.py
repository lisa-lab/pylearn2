from framework.utils import serial #rm
import numpy as N#rm

class DefaultTrainingAlgorithm:
    def __init__(self, batch_size, batches_per_iter , monitoring_batches = - 1, monitoring_dataset = None):
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset, self.monitoring_batches = monitoring_dataset, monitoring_batches
    #

    def train(self, model, dataset):

        for i in xrange(self.batches_per_iter):
            model.learn(dataset, self.batch_size)
        #

        if self.monitoring_dataset:
            self.monitoring_dataset.reset_RNG()
            model.record_monitoring_error(self.monitoring_dataset,batches=self.monitoring_batches,batch_size=self.batch_size)
            print model.error_record[-1]

            """
            self.monitoring_dataset.reset_RNG()
            model.record_monitoring_error(self.monitoring_dataset,batches=self.monitoring_batches,batch_size=self.batch_size)
            print 'objective first time evaluated: '+str(model.error_record[-2][2])
            print 'objective second time evaluated: '+str(model.error_record[-1][2])
            assert model.error_record[-2][2] == model.error_record[-1][2]


            serial.save('/tmp/model.pkl',model) #rm
            model_prime = serial.load('/tmp/model.pkl')#rm
            model_prime.redo_theano()#rm

            for field in dir(model):
                if field not in dir(model_prime):
                    print 'model_prime is missing '+field
                else:
                    old = getattr(model,field)
                    new = getattr(model_prime,field)
                    if 'get_value' in dir(old):
                        print 'checking '+field
                        assert N.all(old.get_value() == new.get_value())

            for field in dir(model_prime):
                if field not in dir(model):
                    print 'model_prime gained '+field


            self.monitoring_dataset.reset_RNG()#rm
            model_prime.record_monitoring_error(self.monitoring_dataset,batches=self.monitoring_batches,batch_size=self.batch_size)
            print 'objective before saving: '+str(model.error_record[-1][2])
            print 'objective after saving: '+str(model_prime.error_record[-1][2])
            assert model_prime.error_record[-1][2] == model.error_record[-1][2]
            print 'succeeded'
            """
        #


        return True
    #
#
