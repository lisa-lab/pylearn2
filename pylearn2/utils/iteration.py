class BatchIterator(object):
    def __init__(self, dataset_size, batch_size, start=0):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.current = start

    def __iter__(self):
        return self

    def next(self):
        if self.current >= self.dataset_size:
            raise StopIteration()
        else:
            data_slice = slice(self.current, self.current + self.batch_size)
            self.current += self.batch_size
            return data_slice

    def reset(self):
        self.current = 0
