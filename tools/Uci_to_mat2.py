import os
import copy
import pandas as pd
import numpy as np
import random as r
from scipy import io


def uci2partial(target_file, start=0, label_index=-1, pl_num=1, pl_percent=100):
    if os.path.exists(target_file):

        df = pd.read_csv(target_file, header=None)
        df_values = df.values
        columns_num = df_values.shape[1]
        data = df_values[:, start:columns_num-1]
        label = df_values[:, label_index]
        unique_label = list(set(label.tolist()))
        class_num = len(unique_label)
        targets = np.zeros((class_num, data.shape[0]))
        for i in range(targets.shape[1]):
            targets[unique_label.index(label[i]), i] = 1
        partial_targets = copy.deepcopy(targets)

        pl_sample_num = int(data.shape[0]*pl_percent/100)
        pl_sample_idx = r.sample(range(0, data.shape[0]), pl_sample_num)

        for i in range(partial_targets.shape[1]):
            if pl_num > 0 and i in pl_sample_idx:
                while True:
                    pls = r.sample(range(0, class_num), pl_num)
                    if unique_label.index(label[i]) not in pls:
                        break
                partial_targets[pls, i] = 1

        io.savemat(target_file, {'data': data.tolist(), 'target': targets, 'partial_target': partial_targets})
    else:
        raise FileNotFoundError(filepath)
    print('finish transform:' + filename)


# uci2partial('abalone', start=1)
# uci2partial('dermatology', start=0)
# uci2partial('ecoli', start=1)
# uci2partial('letter-recognition', start=1)
# uci2partial('pendigits', start=0)
# uci2partial('satimage', start=0)
# uci2partial('segment', start=0)
# uci2partial('vehicle', start=0)
uci2partial('dermatology')
uci2partial('glass')
uci2partial('iris')
uci2partial('satimage')
uci2partial('letter-recognition', start=1, label_index=0)
uci2partial('wine', start=1, label_index=0)

