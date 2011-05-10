
class InstrumentRecord:
    """ A class for recording various instrumented values during learning """

    def __init__(self):
        self.d = {}
        self.initialized = False
        self.in_report = False
    #

    def begin_report(self, examples_seen, batches_seen):
        """ Begins the report for an epoch by specifying how many examples/batches the model has been trained on """
        assert not self.in_report
        self.in_report = True

        self.keys_seen = []

        self.report('examples_seen',examples_seen)
        self.report('batches_seen',batches_seen)
    #

    def report(self, key, value):
        """ Fills in the value for one of the instrumented quantities being tracked. Should be called after begin_report and before end_report """
        assert self.in_report

        if key in self.keys_seen:
            raise Exception("Tried to report the same quantity ("+str(key)+") twice in the same epoch report")
        #

        self.keys_seen.append(key)

        if self.initialized:
            assert key in self.d
        else:
            self.d[key] = []

        self.d[key].append(value)
    #

    def end_report(self):
        self.initialized = True
        self.in_report = False

        for key in self.d:
            if key not in self.keys_seen:
                raise Exception("A quantity ("+str(key)+") was not reported in an epoch report")
            #
        #
    #
#
        

        
