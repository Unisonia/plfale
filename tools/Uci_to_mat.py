import os
import copy
import pandas as pd
import numpy as np
import random as ran
from scipy import io
from sklearn import preprocessing


# r表示候选标签个数
def uci2partial_file(filename, start=0, end_offset=0, label_index=-1, r=1, pl_percent=100):
    assert r >= 0 and 0 <= pl_percent <= 100
    print('start transform:' + filename)
    filepath = os.path.join('../uci', filename + '.csv')
    if os.path.exists(filepath):
        if not os.path.exists(directory):
            os.makedirs(directory)
        target_file = os.path.join(directory, filename + '.mat')
        if os.path.exists(target_file):
            print('Already exist:' + filename)
            return
        df = pd.read_csv(filepath, header=None)
        df_values = df.values
        columns_num = df_values.shape[1]
        data = df_values[:, start:columns_num-end_offset]
        label = df_values[:, label_index]
        unique_label = list(set(label.tolist()))
        class_num = len(unique_label)
        targets = np.zeros((class_num, data.shape[0]))
        for i in range(targets.shape[1]):
            targets[unique_label.index(label[i]), i] = 1
        partial_targets = copy.deepcopy(targets)
        partial_idx = ran.sample(list(range(0, data.shape[0])), int(np.ceil(pl_percent * data.shape[0] / 100)))
        for i in partial_idx:
            labels = list(range(0, class_num))
            labels.remove(unique_label.index(label[i]))
            # 标签随机或者固定
            # c_num = r
            c_num = ran.randint(0, r)
            pls = ran.sample(labels, c_num)
            for j in pls:
                partial_targets[j, i] = 1
        io.savemat(target_file, {'data': data.tolist(), 'target': targets, 'partial_target': partial_targets})
    else:
        raise FileNotFoundError(filepath)
    print('finish transform:' + filename)


def norm_data(data):
    data = preprocessing.scale(data.astype('float64'))
    return data


def load_data(target_file):
    if os.path.exists(target_file):
        mat = io.loadmat(target_file)
        data = mat['data']
        data = norm_data(data)
        targets = mat['target']
        if type(targets) != np.ndarray:
            targets = targets.toarray()
        targets = targets.astype(np.int)
        partial_targets = mat['partial_target']
        if type(partial_targets) != np.ndarray:
            partial_targets = partial_targets.toarray()
        partial_targets = partial_targets.astype(np.int)

    else:
        raise FileNotFoundError(target_file)
    return data, targets, partial_targets


def uci2partial(targets, r=1, pl_percent=30):
    class_num = targets.shape[0]
    true_label_location = np.argmax(targets, axis=0)
    partial_idx = ran.sample(list(range(0, targets.shape[1])), int(np.ceil(pl_percent * targets.shape[1] / 100)))
    partial_targets = copy.deepcopy(targets)
    for idx in partial_idx:
        labels = list(range(0, class_num))
        labels.remove(true_label_location[idx])
        # 标签随机或者固定
        pls = ran.sample(labels, r)
        for j in pls:
            partial_targets[j, idx] = 1
    # io.savemat(target_file, {'data': data.tolist(), 'target': targets, 'partial_target': partial_targets})

    return targets, partial_targets


def uci2partialWithCooccur(targets, co_occur_percent=70):
    class_num = targets.shape[0]
    # true_label_location = np.argmax(targets, axis=0)
    partial_targets = copy.deepcopy(targets)
    for idx in range(class_num):
        same_label_example_idxs = list(np.nonzero(targets[idx, :] == 1)[0])
        co_occur_idxs = set(ran.sample(same_label_example_idxs, int(np.ceil(co_occur_percent * len(same_label_example_idxs) / 100))))
        labels = list(range(0, class_num))
        labels.remove(idx)
        # 标签选取一个标签
        pl = ran.sample(labels, 1)[0]
        for co_occur_idx in co_occur_idxs:
            partial_targets[pl, co_occur_idx] = 1
        # labels.remove(pl)
        other_idxs = set(same_label_example_idxs) - co_occur_idxs
        for other_idx in other_idxs:
            pl = ran.sample(labels, 1)
            partial_targets[pl, other_idx] = 1
    # io.savemat(target_file, {'data': data.tolist(), 'target': targets, 'partial_target': partial_targets})

    return targets, partial_targets


def uci2partial_same(data, targets, r=1, pl_percent=30):
    class_num = targets.shape[0]
    # partial_idx = ran.sample(list(range(0, data.shape[0])), int(np.ceil(pl_percent * data.shape[0] / 100)))
    partial_targets = copy.deepcopy(targets)
    for i in range(targets.shape[0]):
        row = targets[i, :]
        idx = list(np.nonzero(row)[0])
        partial_idxs = ran.sample(idx, int(np.ceil(pl_percent * (len(idx)) / 100)))
        labels = list(range(0, class_num))
        labels.remove(i)
        pls = ran.sample(labels, r)
        for partial_idx in partial_idxs:
            partial_targets[pls, partial_idx] = 1
    # io.savemat(target_file, {'data': data.tolist(), 'target': targets, 'partial_target': partial_targets})

    return data, targets, partial_targets


if __name__ == '__main__':
    pl_nums = [3]
    percents = [10,20,30,40,50,60,70]

    for pl_num in pl_nums:
        for percent in percents:
            for i in range(10):
                # directory = '../mat/uci/'
                directory = '../mat/uci_r' + str(pl_num) + '_p' + str(percent) + '/' + str(i) + '/'

                uci2partial_file('abalone', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('dermatology', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('ecoli', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                # uci2partial_file('letter-recognition', start=1, end_offset=0, label_index=0, r=pl_num, pl_percent=percent)
                # uci2partial_file('pendigits', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('segment', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('vehicle', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('glass', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                # # uci2partial_file('iris', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('leaf', start=2, end_offset=0, label_index=0, r=pl_num, pl_percent=percent)
                # uci2partial_file('satimage', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)

                uci2partial_file('movement_libras', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                # uci2partial_file('lymphography', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
                uci2partial_file('zoo', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
