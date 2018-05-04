import tensorflow as tf
from tensorflow.contrib import rnn


# consturct your DNN model Graph
# The maxnumber of
def mydnn(num_hidden_uni, num_class, f_dim):
    num_hidden = len(num_hidden_uni)
    with tf.variable_scope('mydnn'):
        # Tensor for input layer protocol
        features = tf.placeholder(
            tf.float32, shape=[None, f_dim], name='input_features')
        hid = tf.layers.dense(features, num_hidden_uni[
                             0], activation=tf.nn.relu)
        for i in range(min(num_hidden, 5) - 1):
            hid = tf.layers.dense(hid, num_hidden_uni[
                                 i + 1], activation=tf.nn.relu)
        
        # Unscaled propability of each class
        output_logits = tf.layers.dense(
            hid, num_class, activation=None, name='output_layer')
        return output_logits

