import tensorflow as tf


def placeholder_float32(*shapes, name=None):
    return tf.placeholder(tf.float32, shapes, name=name)


