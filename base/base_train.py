import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, train_data, val_data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.val_data = val_data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        raise NotImplementedError
