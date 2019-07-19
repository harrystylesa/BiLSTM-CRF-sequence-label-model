# !/usr/bin/env python
# coding: utf-8
from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.contrib import rnn


# 双向LSTM分词模型
class Bi_LSTM_crf_with_pretrained_embedding(BaseModel):
    def __init__(self, config):
        super(Bi_LSTM_crf_with_pretrained_embedding, self).__init__(config)
        self.X = None
        self.y = None
        self.time_step_size = config['time_step_size']
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.embedding_vocab_size = config['embedding_vocab_size']
        self.hidden_size = config['hidden_size']
        self.layer_num = config['layer_num']
        self.class_num = config['class_num']
        self.reset_default_graph = config['reset_default_graph']
        self.max_to_keep = config['max_to_keep']
        # self.keep_prob = config['keep_prob']
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        # self.is_training = tf.placeholder(tf.bool, name='training')

        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.int32, [None, self.time_step_size], name='X_input')
            self.y = tf.placeholder(tf.int32, [None, self.time_step_size], name='y_input')
            self.length = tf.reduce_sum(tf.sign(self.X), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

        bilstm_output = self.bi_lstm(self.X)

        with tf.name_scope('softmax'):
            with tf.name_scope('weight'):
                weights = self.weight_variables([self.hidden_size * 2, self.class_num])
                self.variable_summaries(weights)
            with tf.name_scope('bias'):
                biases = self.bias_variables([self.class_num])
                self.variable_summaries(biases)
            with tf.name_scope('linear'):
                preactivate = tf.matmul(bilstm_output, weights) + biases
                print(preactivate.shape)
                tf.summary.histogram('linear', preactivate)
                # preactivate.shape = [batchsize, timestepsize, classnum]
                #   将线性输出经过激励函数，并将输出也用直方图记录下来
            # with tf.name_scope('activation'):
            # activations = tf.nn.softmax(preactivate, name='activation')
            # tf.summary.histogram('activations', activations)

        self.preactive = tf.reshape(preactivate, [self.batch_size, self.time_step_size, self.class_num])
        with tf.name_scope('CRF'):
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.preactive, self.y, self.length)
            self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(
                self.preactive, self.transition_params, self.length)
            self.loss = tf.reduce_mean(-self.log_likelihood)
            summary = tf.summary.scalar("loss", self.loss)

        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar('learning_rate', self.lr)
            self.opt = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        # summaries合并
        self.merged = tf.summary.merge_all()
        print('Finished creating the bi-lstm model.')
        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        pass

    @staticmethod
    def weight_variables(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variables(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell_dropout(self, lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        with tf.name_scope('dropout'):
            tf.summary.scalar('dropout_keep_probablity', self.keep_prob)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        return drop

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            #       计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

        #       计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            #       使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            #       用直方图记录参数的分布
            tf.summary.histogram('histogram', var)

    def bi_lstm(self, X_inputs):
        """build the bi-LSTMs network. Return the y_pred"""

        # batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[self.embedding_vocab_size, self.embedding_size]),
                            trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.embedding_vocab_size, self.embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)

        inputs = tf.nn.embedding_lookup(W, self.X)
        # inputs.shape=[batchsize, timestepsize, embeddingsize]

        # with tf.name_scope('lstm_cell'):

        with tf.name_scope('bi_lstm'):
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell_dropout(self.hidden_size, self.keep_prob) for _ in range(self.layer_num)],
                state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell_dropout(self.hidden_size, self.keep_prob) for _ in range(self.layer_num)],
                state_is_tuple=True)
            initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            # inputs.shape= list of tensor, each tensor's shape = [batch_size, embedding_size]
            inputs = tf.unstack(inputs, self.time_step_size, 1)
            # try:
            outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                         initial_state_fw=initial_state_fw,
                                                         initial_state_bw=initial_state_bw, dtype=tf.float32)
            # except Exception:
            #     #     Old TensorFlow version only returns outputs not states
            #     outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
            #                                            initial_state_fw=initial_state_fw,
            #                                            initial_state_bw=initial_state_bw, dtype=tf.float32)
            outputs = tf.concat(outputs, 1)
            lstm_output = tf.reshape(outputs, [self.batch_size, self.time_step_size, self.hidden_size * 2])
            print(lstm_output.shape)
            # lstm_output.shape=list of tensor, each tensor's shape =[batch_size, 2 * hidden_size]
            lstm_out = tf.reshape(lstm_output, [-1, self.hidden_size * 2])
        return lstm_out  # [-1, hidden_size*2]
