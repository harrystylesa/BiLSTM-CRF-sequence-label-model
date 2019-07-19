from base.base_train import BaseTrain
import time
import tensorflow as tf
import math
import re
import numpy as np
import pandas as pd


class NERTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, val_data, config):
        super(NERTrainer, self).__init__(sess, model, train_data, val_data, config)
        self.word2id = pd.Series.from_csv(self.config['word2id'], encoding='utf-8', header=None)

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
        train_losses = 0.0
        for epoch in range(self.config['epoch_num']):
            # Run 200 steps using the training dataset. Note that the training dataset is
            # infinite, and we resume from where we left off in the previous `while` loop
            # iteration.
            _lr = self.config['learning_rate']
            if epoch > self.config['max_epoch']:
                _lr = _lr * ((self.config['decay']) ** (epoch - self.config['max_epoch']))
            print('EPOCH %d， lr=%g' % (epoch + 1, _lr))
            start_time = time.time()

            val_losses = 0.0
            show_losses = 0.0
            for batch in range(train_batch_num):
                train_batch = self.sess.run(next_element, feed_dict={handle: training_handle})
                X_train = train_batch['data']
                y_train = train_batch['label']
                summary, _, train_loss = self.sess.run([self.model.merged, self.model.opt, self.model.loss],
                                                       feed_dict={self.model.X: X_train, self.model.y: y_train,
                                                                  self.model.lr: _lr, self.model.keep_prob: 0.5,
                                                                  self.model.batch_size: 8})
                train_losses += train_loss
                show_losses += train_loss
                if (batch + 1) % 100 == 0:
                    train_writer.add_summary(summary, global_step=epoch + 1)

            # Run one pass over the validation dataset.
            self.sess.run(validation_iterator.initializer)
            for batch in range(val_batch_num):
                val_batch = self.sess.run(next_element, feed_dict={handle: validation_handle})
                X_val, y_val = val_batch['data'], val_batch['label']
                val_summary, _, val_loss = self.sess.run([self.model.merged, self.model.opt, self.model.loss],
                                                         feed_dict={self.model.X: X_val, self.model.y: y_val,
                                                                    self.model.lr: _lr,
                                                                    self.model.keep_prob: 1.0,
                                                                    self.model.batch_size: 8})
                val_losses += val_loss
                if (batch + 1) % 10 == 0:
                    val_writer.add_summary(summary, global_step=epoch + 1)
            print('\ttraining cost=%g;  valid cost=%g ' % (show_losses / train_batch_num,
                                                           val_losses / val_batch_num))
            if (epoch + 1) % 5 == 0:  # 每 5 个 epoch 保存一次模型
                # self.model.save(self.sess)
                self.model.save(self.sess)
                # save_path = saver.save(self.sess, self.config['ckpt_dir'], global_step=(epoch + 1))
                # print('the save path is ', save_path)
            mean_loss = train_losses / train_batch_num / self.config['epoch_num']
            print('\tcost=%g ' % mean_loss)
            print('Epoch trainining speed=%g s/epoch' % (time.time() - start_time))

    def test(self, testfile):
        rawlines = []
        lines = []
        with open(testfile, "r", encoding='utf-8') as raw:
            for line in raw.readlines():
                rawlines.append(line.strip())
                lines.append(line.strip().replace(" ", ""))
        # print(lines)
        result = self.get_pred(lines, rawlines)
        f = open(testfile + "_ner.txt", 'a', encoding='utf-8')
        f.writelines(result)
        f.close()
        print(result)

    def get_pred(self, lines, rawlines):
        result = []
        for rawline, line in zip(rawlines, lines):
            res = self.ner_word(line, rawline)
            line = "".join(res) + '\n'
            result.append(line)
            print(line)
        return result

    def ner_word(self, sentence, raw_s):
        '''

        :param sentence:
        :param raw_s:
        :return:
        '''
        label_list = ["N", "XX", "XXX", "S-sym", "B-sym", "M-sym", "E-sym", "S-bod", "B-bod", "M-bod", "E-bod", "S-tre",
                      "B-tre", "M-tre", "E-tre", "S-dis", "B-dis", "M-dis", "E-dis", "S-tes", "B-tes", "M-tes", "E-tes",
                      "B-nt", "M-nt", "E-nt"]
        begin_list = ["XX", "XXX", "B-sym", "B-bod", "B-tre", "B-dis", "B-tes", "B-nt"]
        s_list = ["S-tre", "S-sym", "S-bod", "S-dis", "S-tes"]
        end_list = ["E-sym", "E-bod", "E-tre", "E-dis", "E-tes", "E-nt"]
        middle_list = ["M-sym", "M-bod", "M-tre", "M-dis", "M-tes", "M-nt"]
        """首先将一个sentence根据标点划分成多个片段text，然后对每一个片段分词。"""
        not_cuts = re.compile(r'[，。？！；：,\?!;:]')
        labels = []
        start = 0
        # self.word2id = pd.Series.from_csv(self.config['word2id'], encoding='utf-8', header=None)
        id2label = pd.Series.from_csv(self.config['id2label'], encoding='utf-8', header=None)
        for seg_sign in not_cuts.finditer(sentence):
            labels.extend(self.simple_ner(sentence[start:seg_sign.start()], id2label))
            start = seg_sign.end()
        labels.extend(self.simple_ner(sentence[start:], id2label))
        i = 0
        j = 0
        stack = [None for i in labels]
        top = -1
        layer = 0
        result = []
        while i < len(raw_s):
            if raw_s[i] in [' ', '，', '。', '？', '！', '；', '：', ',', '?', '!', ';', ':']:
                result.append(raw_s[i])
            elif j < len(labels):
                if labels[j] == 'N':
                    result.append(raw_s[i])
                elif labels[j] in middle_list:
                    result.append(raw_s[i])
                elif labels[j] in begin_list:
                    top += 1
                    stack[top] = labels[j]
                    if labels[j] in ['XX', 'XXX']:
                        layer = labels[j].count('X')
                    else:
                        layer = 1
                    result.append('[' * layer)
                    result.append(raw_s[i])
                elif labels[j] in end_list:
                    if stack[top] in ['XX', 'XXX']:
                        layer -= 1
                        if layer == 0:
                            top -= 1
                        result.append(raw_s[i])
                        result.append(']' + labels[j][2:])
                    else:
                        top -= 1
                        result.append(raw_s[i])
                        result.append(']' + labels[j][2:])
                elif labels[j] in s_list:
                    if j < len(labels) - 1 and labels[j + 1] in middle_list:
                        result.append('[[' + raw_s[i] + ']' + labels[j][2:])
                        top += 1
                        stack[top] = labels[j]
                    else:
                        result.append('[' + raw_s[i] + ']' + labels[j][2:])
                j += 1
            i += 1
        # print(''.join(result))
        return ''.join(result)

    # @staticmethod
    def text2ids(self, text):
        """把字片段text转为 ids."""
        words = list(text)
        ids = []
        for i in words:
            try:
                id = self.word2id[str(i)]
            except KeyError:
                self.word2id[str(i)] = self.word2id.iloc[-1] + 1
                id = self.word2id[str(i)]
            ids.append(id)
        ids.extend([0] * (32 - len(ids)))  # 短则补全
        ids = np.asarray(ids).reshape([-1, 32])
        return ids

    def simple_ner(self, text, id2label):
        """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
        textlist = []
        tags = []
        if len(text) > 0:
            rest = text
            while True:
                if len(rest) > 32:
                    textlist.append(rest[:min(32, len(rest))])
                    rest = rest[32:]
                else:
                    textlist = [rest]
                    break
            for t in textlist:
                text_len = len(t)
                X_batch = self.text2ids(t)  # 这里每个 batch 是一个样本
                feed_dict = {self.model.X: X_batch, self.model.lr: 1.0, self.model.keep_prob: 1.0,
                             self.model.batch_size: 1}
                batch_pred_sequence = self.sess.run([self.model.batch_pred_sequence], feed_dict)[0][0][
                                      :text_len]  # padding填充的部分直接丢弃
                # print(text_len)
                # print(batch_pred_sequence[0][0][:text_len])
                tags = [id2label[i] for i in batch_pred_sequence]
        # print(tags)
        return tags
