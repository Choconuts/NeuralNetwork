from neural_network import *
from vae_net.vae import *
from lb_cloth_3d.beta_2_cloth_mesh.dataset import AutoEncoderGroundTruth
import numpy as np
from lb_cloth_3d.beta_2_cloth_mesh.mesh_vae import N_DIM, N_HIDDEN, N_CODE, SAVE_PATH, gt
from com.mesh.mesh import Mesh


if __name__ == '__main__':

    # graph
    encode_var_list = []
    decode_var_list = []

    with VarCollector(encode_var_list):
        x_ = placeholder_float32(None, N_DIM, name='input_x')
        keep = placeholder_float32(name='keep_probability')

        # encoding
        mu, sigma = gaussian_MLP_encoder(x_, N_HIDDEN, N_CODE, keep)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z_ = placeholder_float32(None, N_CODE, name='test_z')

    with VarCollector(decode_var_list):
        # decoding
        y = bernoulli_MLP_decoder(z_, N_HIDDEN, N_DIM, keep)

    # train
    interactive_init()

    load_variables(SAVE_PATH, var_list=encode_var_list + decode_var_list)

    test_x = gt.get_test()[1]
    test_z = z.eval(feed_dict={
        x_: test_x,
        keep: 1.0
    })
    verts = y.eval(feed_dict={
        z_: test_z,
        keep: 1.0
    })
    print(test_z)

    for i in range(12):
        Mesh().from_vertices(verts[i].reshape(-1, 3) + gt.template, gt.triangles).save('../save/rebuild_%03d.obj' % i)



