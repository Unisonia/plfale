import os
import copy
import pandas as pd
import numpy as np
import random as ran
from scipy import io


# r表示候选标签个数
def uci2partial(filename, start=0, end_offset=0, label_index=-1, r=1, pl_percent=100):
    assert r > 0 and 0 < pl_percent <= 100
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


pl_nums = [3]
percents = [100]

for pl_num in pl_nums:
    for percent in percents:
        for i in range(10):
            directory = '../mat/uci_r' + str(pl_num) + '_p' + str(percent) + '/' + str(i) + '/'

            uci2partial('abalone', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('dermatology', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('ecoli', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('letter-recognition', start=1, end_offset=0, label_index=0, r=pl_num, pl_percent=percent)
            uci2partial('pendigits', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('segment', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('vehicle', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('glass', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            # uci2partial('iris', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('leaf', start=2, end_offset=0, label_index=0, r=pl_num, pl_percent=percent)
            uci2partial('satimage', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)

            uci2partial('lymphography', start=0, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
            uci2partial('zoo', start=1, end_offset=1, label_index=-1, r=pl_num, pl_percent=percent)
