import json, os
import numpy as np

configure = 'win.json'


def default_path():
    return conf_path('temp')


def find_dir_upwards(dir_name, iter=5):
    if os.path.exists(dir_name) or iter < 0:
        return dir_name
    else:
        return find_dir_upwards(os.path.join('..', dir_name), iter - 1)


def conf_path(key, base=None):
    if base == None:
        base = conf_value('database')

    path = conf_value(key)
    base_dir = get_base(base)

    if path is not None:
        path = os.path.join(base_dir, path)
        return path
    return os.path.join(base_dir, key)


def conf_value(key):
    try:
        with open(conf_json, 'r') as fp:
            obj = json.load(fp)
            if key not in obj:
                return None
            return obj[key]
    except:
        return None


def str3(i):
    if i > 999:
        print("Warning: index overflowed")
    return '%03d' % i


def str4(i):
    if i > 9999:
        print("Warning: index overflowed")
    return '%04d' % i


def str5(i):
    if i > 99999:
        print("Warning: index overflowed")
    return '%05d' % i


def get_base(key=None, max_finding_iter=5):
    if key is None:
        key = conf_value('database')
    base_dir = conf_value(key)
    if base_dir is None:
        base_dir = find_dir_upwards(key, max_finding_iter)
    return base_dir


join = os.path.join
exists = os.path.exists

conf_json = os.path.join(find_dir_upwards('conf'), configure)


def load_json(file):
    if not exists(file):
        return None
    with open(file, 'r') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj, file):
    obj = jsonify(obj)
    with open(file, 'w') as fp:
        json.dump(obj, fp)


def jsonify(root):
    if type(root) == list:
        for i in range(len(root)):
            root[i] = jsonify(root[i])
    elif type(root) == dict:
        for i in root:
            root[i] = jsonify(root[i])
    elif type(root) == np.ndarray:
        return root.tolist()
    return root


def find_last_in_dir(base_dir, file_name_lambda):
    dirs = os.listdir(base_dir)
    i = 0
    while file_name_lambda(i) in dirs:
        i += 1
    return i


if __name__ == '__main__':
    print(conf_json)
    print(conf_path('betas'))
