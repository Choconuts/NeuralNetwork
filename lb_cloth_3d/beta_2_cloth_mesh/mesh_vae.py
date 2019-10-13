from neural_network import *
from vae_net.vae import *
from lb_cloth_3d.beta_2_cloth_mesh.dataset import AutoEncoderGroundTruth
import numpy as np

gt = AutoEncoderGroundTruth()
N_DIM = gt.template.size
N_HIDDEN = 250
N_CODE = 5
KEEP = 0.9

SHOW_STEP = 100
SAVE_STEP = 1000
SAVE_PATH = r'save\model'

if __name__ == '__main__':

    # graph
    encode_var_list = []
    decode_var_list = []

    with VarCollector(encode_var_list):
        x_ = placeholder_float32(None, N_DIM, name='input_x')
        x = placeholder_float32(None, N_DIM, name='target_x')
        keep = placeholder_float32(name='keep_probability')

        # encoding
        mu, sigma = gaussian_MLP_encoder(x_, N_HIDDEN, N_CODE, keep)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    with VarCollector(decode_var_list):
        # decoding
        y = bernoulli_MLP_decoder(z, N_HIDDEN, N_DIM, keep)

    # extend
    # loss
    # marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    #
    # marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    # KL_divergence = tf.reduce_mean(KL_divergence)
    #
    # ELBO = marginal_likelihood - KL_divergence

    # loss = -ELBO
    diff = tf.reduce_mean(tf.square(x - y))
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    loss = diff + KL_divergence

    train_op = AutoDecAdam(1e-3, 100, 0.9).minimize(loss, encode_var_list + decode_var_list)

    # train
    interactive_init()
    show_all_variable()


    def train_step():
        batch_x_ = gt.get_batch(12)[1]
        batch_x = batch_x_
        batch_x_ += (np.random.rand(*batch_x_.shape) - 0.5) * 0.05
        return tf.get_default_session().run(
            (train_op, loss, diff, KL_divergence),
            feed_dict={
                x: batch_x,
                x_: batch_x_,
                keep: KEEP
            }
        )

    def test_step():
        test_x = gt.get_test()[1]
        return tf.get_default_session().run(
            (train_op, loss, diff, KL_divergence),
            feed_dict={
                x: test_x,
                x_: test_x,
                keep: 1.0
            }
        )

    def save_step():
        save_variables(r'D:\Work\NeuralNet\NeuralNetwork\net_tst\save\2', var_list=encode_var_list + decode_var_list)


    for i in range(1, 5001):
        _, _loss, _diff, _KL_divergence = train_step()
        if i % SHOW_STEP == 0:
            print('step %d: training loss \t%g (\t%g,\t%g) ' % (i, _loss, _diff, _KL_divergence), end='')
            _, _loss, _diff, _KL_divergence = test_step()
            print('test loss \t%g (\t%g,\t%g) ' % (_loss, _diff, _KL_divergence))
        if i % SAVE_STEP == 0:
            save_variables(SAVE_PATH, encode_var_list + decode_var_list, global_step=i)

    save_variables(SAVE_PATH, encode_var_list + decode_var_list, write_graph=True)

    # for i in range(5000):
    #     batch_x_ = gt.get_batch(17)[1]
    #     batch_x = batch_x_
    #     batch_x_ += (np.random.rand(*batch_x_.shape) - 0.5) * 0.05
    #     if i % 100 == 0:
    #         test_x = gt.get_test()[1]
    #         _, _loss = tf.get_default_session().run(
    #             (train_op, loss),
    #             feed_dict={
    #                 x: test_x,
    #                 x_: test_x,
    #                 keep: 1.0
    #             }
    #         )
    #         print("step %d, training loss %g" % (i, _loss))
    #         gt.batch_manager.shuffle()
    #
    #     train_op.run(feed_dict={
    #         x: batch_x,
    #         x_: batch_x_,
    #         keep: KEEP
    #     })
