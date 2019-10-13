from tensorflow.examples.tutorials.mnist import input_data
from com.path_helper import *
import tensorflow as tf


def get_mnist():
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    return input_data.read_data_sets(join(get_base("neural_network"), "MNIST/"), one_hot=True)


if __name__ == '__main__':
    print(get_mnist())
