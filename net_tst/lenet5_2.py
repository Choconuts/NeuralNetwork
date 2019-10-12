# -*- coding: utf-8 -*-
"""
Created on XU JING HUI  4-26-2018

@author: root
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
tf.InteractiveSession()

#导入数据，创建一个session对象　，之后的运算都会跑在这个session里
mnist = input_data.read_data_sets("MNIST/",one_hot=True)

#定义一个函数，用于初始化所有的权值 W,这里我们给权重添加了一个截断的正态分布噪声　标准差为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b，这里给偏置加了一个正值0.1来避免死亡节点
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

#定义一个函数，用于构建卷积层，这里strides都是１　代表不遗漏的划过图像的每一个点
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义一个函数，用于构建池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#placceholder 基本都是用于占位符　后面用到先定义
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])                                #将数据reshape成适合的维度来进行后续的计算

#第一个卷积层的定义
W_conv1 = weight_variable([5,5,1,32])
#TODO
W_conv1 = tf.get_collection('variables')[0]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)                #激活函数为relu
h_pool1 = max_pool_2x2(h_conv1)                                        #2x2 的max pooling

#第二个卷积层的定义
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第一个全连接层的定义
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.layers.Dense(
    name='dense_1',
    units=512,
    activation=tf.nn.relu,
)(h_pool2_flat)

#将第一个全连接层　进行dropout　随机丢掉一些神经元不参与运算
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.layers.Dropout(1 - keep_prob)(h_fc1)

#第二个全连接层　分为十类数据　softmax后输出概率最大的数字
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc2 = tf.layers.Dense(
    name='dense_2',
    units=10,
    activation=None,
)(h_fc1_drop)
y_conv = tf.nn.softmax(h_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))        #交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)                                #这里用Ａdam优化器　优化　也可以使用随机梯度下降

correct_predition = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))                                #准确率

# tf.global_variables_initializer().run()                                                                #使用全局参数初始化器　并调用run方法　来进行参数初始化

vs = tf.trainable_variables()
for v in vs:
    print(v)

from tensorflow.python.framework import ops
tf.variable_scope('', reuse=True)
print(ops.GraphKeys.BIASES)
vb = tf.get_collection('variables')
print(vb[0])

with tf.variable_scope("dense_1", reuse=True):
    vb = tf.get_variable('kernel')
    print(vb)

# for i in range(100):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})        #每一百次验证一下准确率
#         print("step %d,training accuracy %g"%(i,train_accuracy))
#
#     train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})                            #batch[0]   [1]　分别指数据维度　和标记维度　将数据传入定义好的优化器进行训练

sv = tf.train.Saver()
# sv.save(tf.get_default_session(), '/Users/choconut/PycharmProjects/DeepLearning/net_tst/save/1')
sv.restore(tf.get_default_session(), '/Users/choconut/PycharmProjects/DeepLearning/net_tst/save/1')

for i in range(100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})        #每一百次验证一下准确率
        print("step %d,training accuracy %g"%(i,train_accuracy))

    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})                            #batch[0]   [1]　分别指数据维度　和标记维度　将数据传入定义好的优化器进行训练


print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))   #开始测试数据




if __name__ == '__main__':
    pass