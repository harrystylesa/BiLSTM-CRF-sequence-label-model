import numpy as np
from data_preprocessor import preprocessor


class DataGenerator:
    def __init__(self, config, isval):
        self.config = config
        # load data here
        batch_size = self.config['batch_size']
        if isval:
            file_name = self.config['val']
            epoch = 1
            shuffle_size = batch_size
        else:
            file_name = self.config['train']
            epoch = self.config['epoch_num']
            shuffle_size = self.config['shuffle_size']
        self.batch_set = preprocessor.read_tfrecords(file_name, epoch, batch_size, shuffle_size)

    def next_batch(self):
        batch = self.iterator.get_next()
        X_input = batch['data']
        y = batch['label']
        yield X_input, y
