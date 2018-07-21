# Basic implementation of an Elman RNN

import tensorflow as tf
import numpy as np

def elman_rnn(num_context, num_hidden, num_classes):
    # Build the Elman RNN model

    params = {}

    # Input layer
    with tf.name_scope('input'):
        with tf.name_scope('input_data'):
            data = tf.placeholder(tf.int32, name='data')
        with tf.name_scope('x'):
            num_inputs = tf.shape(data)[0]
            wx = tf.Variable(name='wx', initial_value=initialize(num_context, num_hidden))
            params['wx'] = wx
            x = tf.Variable(name='x', initial_value=initialize(num_inputs, num_context))

    # Recurrence layer
    with tf.name_scope('recurrence'):
        with tf.name_scope('recurrence_init'):
            h = tf.Variable(name='h', initial_value=np.zeros((1, num_hidden)))
            h_0 = tf.reshape(h, (1, num_hidden))
            params['h'] = h
            s_0 = tf.constant(np.matrix([0.] * num_classes))
            y_0 = tf.constant(0, 'int64')
        with tf.name_scope('hidden'):
            wh = tf.Variable(name='wh', initial_value=initialize(num_hidden, num_hidden))
            bh = tf.Variable(name='bh', initial_value=np.zeros((1, num_hidden)))
            params['wh'] = wh
            params['bh'] = bh
        with tf.name_scope('classes'):
            w = tf.Variable(name='w', initial_value=initialize(num_hidden, num_classes))
            b = tf.Variable(name='b', initial_value=np.zeros((1, num_classes)))
            params['w'] = w
            params['b'] = b
        h, s, y = tf.scan(recurrence, x, initializer=(h_0, s_0, y_0))

    # Output layer
    with tf.name_scope('output'):
        with tf.name_scope('target_data'):
            target = tf.placeholder(tf.float64, name='target')
        with tf.name_scope('probabilities'):
            s = tf.squeeze(s)
        with tf.name_scope('outcomes'):
            y = tf.squeeze(y)
        with tf.name_scope('loss'):
            loss = -tf.reduce_sum(target * tf.log(tf.clip_by_value(s, 1e-20, 1.0)))

    return loss

@staticmethod
def initialize(*shape):
    return 0.001 * np.random.uniform(-1., 1., shape)

def recurrence(self, old_state, x_t):
    h_t, s_t, y_t = old_state
    x = tf.reshape(x_t, (1, self.num_context))
    input_layer = tf.matmul(x, self.wx)
    hidden_layer = tf.matmul(h_t, self.wh) + self.bh
    h_t_next = input_layer + hidden_layer
    s_t_next = tf.nn.softmax(tf.matmul(h_t_next, self.w) + self.b)
    y_t_next = tf.squeeze(tf.argmax(s_t_next, 1))
    return h_t_next, s_t_next, y_t_next

    