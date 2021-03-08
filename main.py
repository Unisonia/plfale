# encoding:utf-8
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from ecoc import *
from tools import Tools
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from learner import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from knn.PLKNN import PLKNN
from plsvm.plsvm import PLSVM
from clpl.CLPL import CLPL
import time
import numpy as np
import os
import scipy.io as sio
import platform
import copy


def draw_matrix(directory, ecoc, split_ts_labels, pre_label_matrix, name, i, acc):
    # Tools.dump_mat(ecoc.original_matrix, os.path.join(directory, 'original_matrix_' + str(ite) + '.csv'), fmt='%d', delimiter=',')
    # Tools.dump_mat(ecoc.coding_matrix, os.path.join(directory, 'code_matrix_' + str(ite) + '.csv'), fmt='%d', delimiter=',')
    # Tools.draw_mat(ecoc.original_matrix, 'original_matrix_', ecoc.classify_scores, os.path.join(directory, str(i) + '_' + name + '_original_matrix.png'), zoom=2,
    #                label_x='Num: ' + str(len(ecoc.models)) + '  average_acc: ' + str(np.mean(ecoc.classify_scores)))
    # Tools.draw_mat(ecoc.performance_matrix, 'performance_matrix', list(range(ecoc.performance_matrix.shape[1])), os.path.join(directory, str(i) + '_' + name + '_performance.png'), label_y='column')
    # all_labels = range(0, split_ts_labels.shape[0])
    # Tools.draw_mat(ecoc.distance_value, 'distance_value', np.dot(all_labels, split_ts_labels), os.path.join(directory, str(i) + '_' + name + '_distance.png'), label_y='column')
    # if ecoc.common_value is not None:
    #     Tools.draw_mat(ecoc.common_value, 'common_value', list(range(ecoc.common_value.shape[1])), os.path.join(directory, str(i) + '_' + name + '_match_value.png'), label_y='column')
    # if ecoc.error_value is not None:
    #     Tools.draw_mat(ecoc.error_value, 'error_value', list(range(ecoc.error_value.shape[1])), os.path.join(directory, str(i) + '_' + name + '_error_value.png'), label_y='column')
    # if ecoc.bin_pre is not None:
    #     Tools.draw_mat(ecoc.bin_pre, 'bin_pre', list(range(ecoc.bin_pre.shape[1])), os.path.join(directory, str(i) + '_' + name + '_bin_pre.png'), label_y='column')
    Tools.draw_mat(ecoc.coding_matrix, 'coding_matrix', ecoc.classify_scores, os.path.join(directory, str(i) + '_' + name + '_code_matrix.png'), zoom=2,
                   label_x='Num: ' + str(len(ecoc.models)) + '  average_acc: ' + str(np.mean(ecoc.classify_scores)))
    if ecoc.co_exist is not None:
        Tools.draw_mat(ecoc.co_exist, 'co_exist', list(range(ecoc.co_exist.shape[1])), os.path.join(directory, str(i) + '_' + name + '_co_exist.png'), label_y='column')
    # 行汉明距离
    hamming_distance = np.zeros((ecoc.coding_matrix.shape[0], ecoc.coding_matrix.shape[0]))
    for m in range(ecoc.coding_matrix.shape[0]):
        r1 = ecoc.coding_matrix[m, :]
        for n in range(m+1, ecoc.coding_matrix.shape[0]):
            r2 = ecoc.coding_matrix[n, :]
            hamming_distance[m, n] = hamming_distance[n, m] = np.sum((r1 - r2) != 0)
    Tools.draw_mat(hamming_distance, 'row_hamming_distance', list(range(hamming_distance.shape[1])), os.path.join(directory, str(i) + '_' + name + '_row_hamming_distance.png'), label_y='column')
    data_class = np.array(range(split_ts_labels.shape[0]))
    ts_vector = np.dot(data_class, split_ts_labels)
    pre_vector = np.dot(data_class, pre_label_matrix)
    confusion = confusion_matrix(ts_vector.tolist(), pre_vector.tolist(), data_class)
    Tools.draw_mat(confusion.astype(np.int), 'confusion', list(range(ecoc.coding_matrix.shape[1])), os.path.join(directory, str(i) + '_' + name + '_confusion.png'), label_x='acc:'+str(acc))


def draw_hist(directory, myList, name_list, Title, Xlabel, Ylabel, Ymin, Ymax, file_name):
    plt.figure(figsize=[2*np.ceil(len(myList)), np.ceil(len(myList))])
    # name_list = list(range(len(myList)))
    rects = plt.bar(range(len(myList)), myList, color='rgby')
    # X轴标题
    index = list(range(len(myList)))
    # index = [float(c)+0.4 for c in range(len(myList))]
    plt.ylim(top=max(myList), bottom=Ymin)
    plt.xticks(index, name_list)
    plt.ylabel(Ylabel) #X轴标签
    plt.xlabel(Xlabel)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, str(height), ha='center', va='bottom')
    plt.title(Title)
    # plt.show()
    plt.savefig(os.path.join(directory, file_name + '.png'))
    plt.cla()


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def clear_dir(directory):
    if os.path.exists(directory):
        filelist = os.listdir(directory)
        for f in filelist:
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
    else:
        os.makedirs(directory)


# 归一化数据集
def norm_data(data):
    data = preprocessing.scale(data)
    # if np.min(data) >= 0:
    # max_abs_scaler = preprocessing.MinMaxScaler()
    # data = max_abs_scaler.fit_transform(data)
    # else:
    #     max_abs_scaler = preprocessing.MaxAbsScaler()
    #     data = max_abs_scaler.fit_transform(data)
    return data


def isWindowsSystem():
    return 'Windows' in platform.system()


def isLinuxSystem():
    return 'Linux' in platform.system()


def run():
    k_fold = 10
    draw_flag = 0
    make_dir(directory)
    if draw_flag:
        clear_dir(directory)
    real_datasets = [
                    # "BirdSong.mat"
                    # , "lost.mat"
                    "MSRCv2.mat"
                    # "Soccer Player.mat"
                    # , "Yahoo! News.mat"
                    # , "FG-NET.mat"
                     ]
    # real_datasets = []
    uci_datasets = [
        # "ecoli.mat"
        # , "vehicle.mat"
        # , "segment.mat"
        # , "dermatology.mat"
        # , "glass.mat"
        # # # , "iris.mat"
        # , "satimage.mat"
        # , "pendigits.mat"
        # #  "abalone.mat"
        # , "letter-recognition.mat"
        # , "leaf.mat"
        # , "lymphography.mat"
        # , "zoo.mat"
    ]
    uci_datasets = []
    # base_learners = [SVM, Bayes, DecisionTree, KNeighbors, RandomForest]
    base_learners = [RandomForest]
    combine_threshold = 0.5
    pool = ThreadPoolExecutor(max_workers=13)
    repetitions = 10
    means = {}
    for repetition in range(repetitions):
        for base_learner in base_learners:
            base_model = {
                            # "Greedy-11": {model_name: Greedy11.GreedyEcoc,
                            #          param_name : {"classifier": base_learner, "svm_param": '-t 2 -c 1 -q', "col_mutiple": 10, "max_iter": 2000}}
                          "Greedy-12": {model_name: Greedy12.GreedyEcoc,
                                       param_name: {"classifier": base_learner, "svm_param": '-t 2 -c 1 -q', "col_mutiple": 10, "max_iter": 2000, "combine_threshold": combine_threshold}}
                           # "Greedy-13": {model_name: Greedy13.GreedyEcoc,
                           #             param_name: {"classifier": base_learner, "svm_param": '-t 2 -c 1 -q', "col_mutiple": 10, "max_iter": 2000}}
                          , "PLSVM": {model_name: PLSVM, param_name: {}}
                          , "PLKNN": {model_name: PLKNN, param_name: {"k": 10}}
                          , "CLPL": {model_name: CLPL, param_name: {}}
                          , "PL-ECOC": {model_name: Rand.RandPLECOC, param_name: {"classifier": base_learner, "svm_param": '-t 2 -c 1 -q'}}
                          }
            accuracies = {}
            code_length = {}
            fscores = {}
            skf = StratifiedKFold(n_splits=k_fold)
            file_paths = {}
            for uci_dataset in uci_datasets:
                file_paths[uci_dataset] = os.path.join(base_path, uci_base_path, str(repetition), uci_dataset)
            for real_dataset in real_datasets:
                file_paths[real_dataset] = os.path.join(base_path, real_dataset)
            for dataset, path in file_paths.items():
                all_task = []
                print("start:" + dataset)
                mat = sio.loadmat(path)
                data = mat['data']
                pl_labels = mat['partial_target']
                if type(pl_labels) != np.ndarray:
                    pl_labels = pl_labels.toarray()
                pl_labels = pl_labels.astype(np.int)
                true_labels = mat['target']
                if type(true_labels) != np.ndarray:
                    true_labels = true_labels.toarray()
                true_labels = true_labels.astype(np.int)
                data = norm_data(data)
                labels = list(range(0, true_labels.shape[0]))
                y = np.dot(labels, true_labels)
                print(dataset + " dimension: " + str(data.shape) + " label num: " + str(pl_labels.shape[0]))
                i = 0
                for tr_index, ts_index in skf.split(data, y):
                    i = i + 1
                    tr_data, ts_data = data[tr_index, :], data[ts_index, :]
                    tr_label, ts_label = pl_labels[:, tr_index], true_labels[:, ts_index]
                    draw_hist_flag = True
                    all_task.extend([pool.submit(train_and_predict, *(tr_data, ts_data, tr_label, ts_label, k, v, i)) for k, v in base_model.items()])
                for future in as_completed(all_task):
                    [model, lables_matrix, ts_label, acc, fscore, k, i] = future.result()
                    key = k + " " + dataset
                    print(key + " time " + str(i) + " acc:" + str(acc) + " fsocre:" + str(fscore))
                    if key in accuracies:
                        acc_list = accuracies[key]
                        acc_list.append(acc)
                    else:
                        acc_list = list()
                        acc_list.append(acc)
                        accuracies[key] = acc_list
                    if key in fscores:
                        score_list = fscores[key]
                        score_list.append(fscore)
                    else:
                        score_list = list()
                        score_list.append(fscore)
                        fscores[key] = score_list
                    if model.is_ecoc:
                        if key in code_length:
                            len_list = code_length[key]
                            len_list.append(model.codingLength)
                        else:
                            len_list = list()
                            len_list.append(model.codingLength)
                            code_length[key] = len_list
                    if draw_flag and model.is_ecoc:
                        # pool.submit(draw_matrix, *(directory, model, ts_label, lables_matrix, key, i, acc))
                        draw_matrix(directory, model, ts_label, lables_matrix, key, i, acc)
                    if draw_flag and draw_hist_flag:
                        # pool.submit(draw_hist, *(directory, model.num_list, model.name_list, 'class_distribution', 'class', 'number', 0,
                        #           np.ceil(max(model.num_list)/1000)*1000, str(i) + '_' + key + '_partial_dist'))
                        draw_hist(directory, model.num_list, model.name_list, 'class_distribution', 'class', 'number', 0,
                                            np.ceil(max(model.num_list)/1000)*1000, str(i) + '_' + key + '_partial_dist')
                        draw_hist_flag = False
                for k, v in accuracies.items():
                    if k in means:
                        lst = means[k]
                        lst.append(np.mean(v))
                    else:
                        lst = list()
                        lst.append(np.mean(v))
                        means[k] = lst
                    print(k + " acc:" + str(v))
                    print(k + "mean fscore:" + str(np.mean(fscores[k])) + " fscore:" + str(fscores[k]))
                    print(k + " mean: " + str(np.round(np.mean(v), 4)) + " std:" + str(np.std(v)) + " max:" + str(max(v)) + ' min:' + str(min(v)) )
                # print(dataset + " dimension: " + str(data.shape) + " label num: " + str(pl_labels.shape[0]))
                # print("finish:" + dataset)
                # print(str(base_learner))
    f = open(os.path.join(directory, 'result' + time_offset + '.txt'), 'a')
    for k, v in means.items():
        f.write(k + " acc:" + str(v) + '\r\n')
        print(k + " acc:" + str(v) + '\r\n')
        # f.write(k + "mean fscore:" + str(np.mean(fscores[k])) + " fscore:" + str(fscores[k]) + '\r\n')
        f.write(k + " mean: " + str(np.round(np.mean(v), 4)) + " std:" + str(np.std(v)) + " max:" + str(max(v)) + ' min:' + str(min(v)) + '\r\n')
        print(k + " mean: " + str(np.round(np.mean(v), 4)) + " std:" + str(np.std(v)) + " max:" + str(max(v)) + ' min:' + str(min(v)) + '\r\n')
    # f.write(str(base_learner))
    f.write(' \r\n')
    f.write(' \r\n')
    f.close()
    print(uci_base_path)


def train_and_predict(tr_data, ts_data, tr_label, ts_label, k, v, i):
    model = v[model_name](v[param_name])
    model.fit(tr_data, tr_label)
    lables_matrix, acc, fscore = model.score(ts_data, ts_label)
    acc = np.round(acc, 4)
    fscore = np.round(fscore, 4)
    return model, lables_matrix, ts_label, acc, fscore, k, i


time_offset = str(time.time())
model_name = "model"
param_name = "params"
base_path = "mat"
uci_base_path = "uci_r1_p30"
day = time.strftime("%Y%m%d")
isWindows = isWindowsSystem()
if isWindows:
    directory = os.path.join('E:/ecoc_dump', day)
else:
    directory = os.path.join('/home/root/PLECOC/dump', day)
if __name__ == '__main__':
    run()
