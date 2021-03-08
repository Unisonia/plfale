import random
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


def tr_ts_split_idx(data, ts_size_percent=0.1):
    random.seed(np.random.randint(0, 100))
    data_size = data.shape[0]
    ts_size = int(np.ceil(data_size * ts_size_percent))
    tr_size = data_size - ts_size
    data_idx = list(range(data_size))
    tr_idx = random.sample(data_idx, tr_size)
    ts_idx = list(set(data_idx) - set(tr_idx))
    return tr_idx, ts_idx


def tr_ts_split(data, ts_size_percent=0.1):
    random.seed(np.random.randint(0, 100))
    data_size = data.shape[0]
    ts_size = int(np.ceil(data_size * ts_size_percent))
    tr_size = data_size - ts_size
    data_idx = list(range(data_size))
    tr_idx = random.sample(data_idx, tr_size)
    ts_idx = list(set(data_idx) - set(tr_idx))
    return data[tr_idx, :], data[ts_idx, :]


def tr_ts_data_label_split(data, label, percent=0.1):
    random.seed(np.random.randint(0, 100))
    data_size = data.shape[0]
    ts_size = int(np.ceil(data_size * percent))
    tr_size = data_size - ts_size
    data_idx = list(range(data_size))
    tr_idx = random.sample(data_idx, tr_size)
    ts_idx = list(set(data_idx) - set(tr_idx))
    return data[tr_idx, :], data[ts_idx, :], label[:, tr_idx], label[:, ts_idx]


def dump_mat(mat, file_path, **param):
    with open(file_path, 'wb') as file:
        np.savetxt(file, np.round(mat, 4), **param)

    # print('Copy finish. time cost: {:>10.2f} hours'.format((time.time()-start_time)/3600))


def draw_mat(mat, title, file_path, zoom=1, label_x='accuracy', label_y='class'):
    plt.figure(figsize=[np.ceil(mat.shape[1]/zoom), np.ceil(mat.shape[0]/zoom)])
    sns.heatmap(np.round(mat, 4), annot=True, cbar=False, cmap='Blues')
    plt.yticks([i + 0.5 for i in list(range(mat.shape[0]))], list(range(mat.shape[0])), rotation=0)
    plt.ylabel(label_y)
    plt.xticks([i + 0.5 for i in list(range(mat.shape[1]))], list(range(mat.shape[0])), rotation=0)
    plt.xlabel(label_x)
    plt.tick_params(length=0)
    plt.title(title)
    plt.savefig(os.path.join(file_path))
    # plt.show()
    plt.close()
