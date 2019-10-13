from functools import wraps
import tensorflow as tf


def collect_variable():

    def decorator(f):

        @wraps(f)
        def wrapper(*args, var_list: list=None,  **kwargs):
            vs = set(tf.trainable_variables())

            y = f(*args, **kwargs)

            vs2 = set(tf.trainable_variables())

            if var_list is not None:
                var_list.extend(vs2- vs)

            return y

        return wrapper

    return decorator


def collect_trainable_trainable(f):

    @wraps(f)
    def wrapper(*args, var_list: list=None,  **kwargs):
        vs = set(tf.trainable_variables())

        y = f(*args, **kwargs)

        vs2 = set(tf.trainable_variables())

        if var_list is not None:
            var_list.extend(vs2- vs)

        return y

    return wrapper
