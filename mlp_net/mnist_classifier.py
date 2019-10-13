from mlp_net.mlp import MLP
from neural_network import *

n_output = 7366 * 3
n_hidden = [20]


def mlp_on_mnist(x: tf.Tensor, y_true: tf.Tensor, keep_prob):
    return MLP(y_true.shape[1], [128, 20]).predict(x, keep_prob)


if __name__ == '__main__':
    # graph
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    x_input = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y")

    var_list = []
    with VarCollector(var_list):
        y = mlp_on_mnist(x_input, y_true, keep_prob)

    # extend
    cross_entropy = tf.reduce_mean(tf.square(y_true - y))
    train_step = AutoDecAdam(1e-3, 100, 0.8).minimize(cross_entropy, var_list)

    # train
    interactive_init()
    show_all_variable()
    mnist = get_mnist()

    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = cross_entropy.eval(feed_dict={x_input:batch[0], y_true:batch[1], keep_prob:1.0})
            print("step %d,training accuracy %g"%(i,train_accuracy))

        train_step.run(feed_dict={x_input:batch[0],y_true:batch[1],keep_prob:0.5})

    save_variables(r'D:\Work\NeuralNet\NeuralNetwork\net_tst\save\2', var_list=var_list)

