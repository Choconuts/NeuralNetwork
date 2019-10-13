import tensorflow as tf


def placeholder_float32(*shapes):
    return tf.placeholder(tf.float32, shapes)


