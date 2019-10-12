from functools import wraps
import tensorflow as tf

def collect_trainable_variable():

    def need_login(f):

        @wraps(f)
        def wrapper(*args, var_list: list=None,  **kwargs):
            vs = set(tf.trainable_variables())

            y = f(*args, **kwargs)

            vs2 = set(tf.trainable_variables())

            if var_list is not None:
                var_list.extend(vs2- vs)

            return y

        return wrapper

    return need_login