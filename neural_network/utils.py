import tensorflow as tf


def show_all_variable():
    vs = tf.trainable_variables()
    for v in vs:
        print(v)


def interactive_init():
    tf.InteractiveSession()
    tf.global_variables_initializer().run()                                                                #使用全局参数初始化器　并调用run方法　来进行参数初始化


def save_variables(path, var_list, sess=None, global_step=None, write_graph=False):
    if sess is None:
        sess = tf.get_default_session()

    sv = tf.train.Saver(var_list)
    sv.save(sess, path, global_step=global_step, write_meta_graph=write_graph)


def load_variables(path, var_list, sess=None, global_step=None):
    if sess is None:
        sess = tf.get_default_session()
    if global_step is not None:
        path += '-%d' % global_step

    sv = tf.train.Saver(var_list)
    sv.restore(sess, path)


class VarCollector:

    def __init__(self, var_list, trainable_only=True):
        self.var_list = var_list
        self.vs = set()
        self.trainable_only = trainable_only

    def collect(self):
        if self.trainable_only:
            return set(tf.trainable_variables())
        else:
            return set(tf.global_variables())

    def __enter__(self):
        self.vs = self.collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        vs = self.collect()

        if type(self.var_list) == list:
            self.var_list.extend(vs - self.vs)
        if type(self.var_list) == set:
            self.var_list += vs - self.vs


class AutoDecAdam:

    def __init__(self, initial_learning_rate, decay_steps=0, decay_rate=0.9):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, loss, var_list=None, *args, **kwargs):
        with tf.variable_scope('auto_dec_adam'):
            global_step = tf.Variable(0, trainable=False)
            if self.decay_steps > 0 and 1 > self.decay_rate > 0:
                rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                  global_step=global_step,
                                                  decay_steps=self.decay_steps, decay_rate=self.decay_rate)
            else:
                rate = self.initial_learning_rate
        return tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step, var_list=var_list)

    def minimize(self, loss, var_list=None):
        return self.__call__(loss, var_list)


