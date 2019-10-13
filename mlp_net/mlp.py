import tensorflow as tf
import numpy as np
from neural_network import *


def full_conn_layer(input_layer, output_size, keep_prob=None, activate=None, name=None):
    dim = 1
    input_shape = np.shape(input_layer)
    for i in range(1, len(input_shape)):
        dim *= int(input_shape[i])      # 拉成向量
    if len(input_shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, dim])
    shape = [dim, output_size]
    w_fc = tf.Variable(tf.truncated_normal(shape, stddev=0.01))  # 权值
    b_fc = tf.Variable(tf.constant(0.01, shape=[output_size]))   # 偏置
    d_fc = tf.matmul(input_layer, w_fc) + b_fc
    if keep_prob is not None:
        d_fc = tf.nn.dropout(d_fc, keep_prob)
    if activate is not None:
        h_fc = activate(d_fc)        # 激活函数
    else:
        h_fc = d_fc
    h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob, name=name)
    return h_fc


class MLP:

    def __init__(self, outputs, hidden_layers):
        self.outputs_dim = outputs

        if type(hidden_layers) == int:
            hidden_layers = [hidden_layers]

        from collections import Iterable
        assert isinstance(hidden_layers, Iterable)

        self.hidden_layers = hidden_layers

    def __call__(self, x, keep_prob, *args, **kwargs):
        full = x
        dropout = tf.keras.layers.Dropout(1 - keep_prob)
        for h in self.hidden_layers:
            if h == 0:
                continue
            dense = tf.keras.layers.Dense(
                units=h,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.orthogonal,
                # kernel_regularizer=tf.keras.regularizers.l1_l2,
                activation=tf.nn.relu,
            )
            full = dense(full)
            full = dropout(full)
        dense = tf.keras.layers.Dense(
            units=self.outputs_dim,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.orthogonal,
            # kernel_regularizer=tf.keras.regularizers.l1_l2,
        )
        output = dense(full)
        return output

    def predict(self, x, keep_prob=1):
        return self.__call__(x, keep_prob)


if __name__ == '__main__':
    print(list(1))
