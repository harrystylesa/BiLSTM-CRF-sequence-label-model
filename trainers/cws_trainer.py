from base.base_train import BaseTrain
import time
import tensorflow as tf
import math


class CWSTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, val_data, config):
        super(CWSTrainer, self).__init__(sess, model, train_data, val_data, config)

    def train(self):
        saver = self.model.saver
        # tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.config['log_dir'] + "/train", self.sess.graph)
        val_writer = tf.summary.FileWriter(self.config['log_dir'] + "/validation")

        # Define training and validation datasets with the same structure.
        training_dataset = self.train_data.batch_set
        validation_dataset = self.val_data.batch_set

        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
        next_element = iterator.get_next()

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = self.sess.run(training_iterator.string_handle())
        validation_handle = self.sess.run(validation_iterator.string_handle())

        train_batch_num = math.ceil(0.1 * self.config['train_sample_size'] / self.config['batch_size'])
        val_batch_num = math.ceil(0.1 * self.config['val_sample_size'] / self.config['batch_size'])
        # Loop forever, alternating between training and validation.
        for epoch in range(self.config['epoch_num']):
            # Run 200 steps using the training dataset. Note that the training dataset is
            # infinite, and we resume from where we left off in the previous `while` loop
            # iteration.
            _lr = self.config['learning_rate']
            if epoch > self.config['max_epoch']:
                _lr = _lr * ((self.config['decay']) ** (epoch - self.config['max_epoch']))
            print('EPOCH %d， lr=%g' % (epoch + 1, _lr))
            start_time = time.time()
            train_losses = 0.0
            val_losses = 0.0
            show_losses = 0.0
            for batch in range(train_batch_num):
                train_batch = self.sess.run(next_element, feed_dict={handle: training_handle})
                X_train = train_batch['data']
                y_train = train_batch['label']
                summary, _, train_loss = self.sess.run([self.model.merged, self.model.opt, self.model.loss],
                                                       feed_dict={self.model.X: X_train, self.model.y: y_train,
                                                                  self.model.lr: _lr, self.model.keep_prob: 0.5})
                train_losses += train_loss
                show_losses += train_loss
                if (batch + 1) % 100 == 0:
                   train_writer.add_summary(summary, global_step=epoch)

            # Run one pass over the validation dataset.
            self.sess.run(validation_iterator.initializer)
            for batch in range(val_batch_num):
                val_batch = self.sess.run(next_element, feed_dict={handle: validation_handle})
                X_val, y_val = val_batch['data'], val_batch['label']
                val_summary, _, val_loss = self.sess.run([self.model.merged, self.model.opt, self.model.loss],
                                                         feed_dict={self.model.X: X_val, self.model.y: y_val,
                                                                    self.model.lr: _lr,
                                                                    self.model.keep_prob: 1.0})
                val_losses += val_loss
                if (batch + 1) % 100 == 0:
                    val_writer.add_summary(summary, global_step=epoch)
            print('\ttraining cost=%g;  valid cost=%g ' % (show_losses / train_batch_num,
                                                           val_losses / val_batch_num))
            if (epoch + 1) % 5 == 0:  # 每 3 个 epoch 保存一次模型
                self.model.save(self.sess)
                # save_path = saver.save(self.sess, self.config['ckpt_dir'], global_step=(epoch + 1))
                # print('the save path is ', save_path)
            mean_loss = train_losses / train_batch_num / self.config['epoch_num']
            print('\tcost=%g ' % mean_loss)
            print('Epoch trainining speed=%g s/epoch' % (time.time() - start_time))
