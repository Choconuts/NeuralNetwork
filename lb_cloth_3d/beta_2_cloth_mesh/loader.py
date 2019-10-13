from com.path_helper import *
import numpy as np, json
from com.mesh.smooth import *


def txt_to_array_fixed_width(txt_file, dtype, width):
    with open(txt_file, 'r') as fp:
        s = ' '
        data = []
        while s:
            s = fp.readline()
            if len(s) < width:
                continue
            values = s.split(' ')
            tri = []
            for i in range(width):
                vi = values[i]
                tri.append(vi)
            data.append(tri)
        data = np.array(data, dtype)
    return data


def txt_to_array(txt_file, dtype):
    with open(txt_file, 'r') as fp:
        s = None
        data = []

        def read_value():
            s = fp.readline()
            if len(s) < 1:
                return None
            values = s.split(' ')
            return values[-1]

        while s is None:
            s = read_value()
        w = int(float(s))
        s = read_value()
        h = int(float(s))

        data = np.zeros((h, w), dtype)
        for i in range(h):
            for j in range(w):
                s = read_value()
                data[i, j] = s
        return data


def txt_to_array_fixed_shape(txt_file, dtype, shape, row_splitter=' '):
    with open(txt_file, 'r') as fp:
        s = None
        data = []

        queue = []

        def read_value():
            if len(queue) > 0:
                x = queue[0]
                queue.pop(0)
                return x
            s = fp.readline()
            if len(s) < 1:
                return None
            values = s.split(row_splitter)
            for v in values:
                if len(v) > 0:
                    queue.append(v)
            return read_value()

        n = 1
        for dim in shape:
            n *= dim
        ii = 0
        for i in range(n):
            while s is None:
                s = read_value()
                ii += 1
                if ii > 1000:
                    return None
            ii = 0
            data.append(s)
            s = None

        return np.array(data, dtype).reshape(shape)

