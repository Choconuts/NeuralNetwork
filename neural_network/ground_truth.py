import os
import numpy as np
import random


class GroundTruth:

    def get_batch(self, size):
        return [[0], [0]]

    def get_test(self):
        return [[0], [0]]

    def load(self, gt_file):
        return self

    def save(self, gt_file):
        return self


class BatchManager:

    def __init__(self, max_num, cut):
        self.max_num = max_num
        self.pointer = 0
        self.train_cut = cut
        self.element = np.linspace(0, max_num - 1, max_num).astype('i')
        self.auto_shufle = False
        self.epoched = False

    def cut(self, new_cut):
        if self.train_cut > new_cut:
            print('warning: smaller new cut!')
        self.train_cut = new_cut
        if self.pointer >= new_cut:
            self.pointer = new_cut - 1
        return self

    def shuffle(self):
        random.shuffle(self.element[0:self.train_cut])
        return self

    def shuffle_all(self):
        random.shuffle(self.element)
        return self

    def get_batch_samples(self, size):
        tmp_ptr = self.pointer
        self.pointer += size
        if self.pointer >= self.train_cut:
            self.epoched = True
        self.pointer %= self.train_cut
        return self.get_range_samples(tmp_ptr, tmp_ptr + size)

    def get_range_samples(self, start, end):
        res = []
        for i in range(start, end):
            ii = i % self.train_cut
            res.append(self.element[ii])
        return res

    def get_test_samples(self):
        return self.element[self.train_cut:]


def ids_2_batch(sample_from_id, ids):
    """
    each sample should be iterable through batch attributes
    :param sample_from_id:
    :param ids:
    :return:
    """
    batch = []
    for i in ids:
        sample = sample_from_id(i)
        while len(batch) < len(sample):
            batch.append([])
        for k, val in enumerate(sample):
            batch[k].append(val)

    for i, _ in enumerate(batch):
        batch[i] = np.array(batch[i])

    return batch


if __name__ == '__main__':
    a = np.linspace(0, 10, 11).astype('i')
    random.shuffle(a)
    print(a)

