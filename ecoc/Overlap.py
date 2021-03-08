import numpy as np
import collections
import math
import pandas as pd
import copy
from ecoc.BasePLECOC import BasePLECOC
from svmutil import *
from sklearn.metrics import accuracy_score


class OverlapEcoc(BasePLECOC):

    def __init__(self,  params):
        BasePLECOC.__init__(self, params)
        self.max_iter = self.params.get("max_iter", 2000)
        self.class_num = None
        self.codingLength = None
        self.threshold = None
        self.coding_matrix = None
        self.models = None
        self.performance_matrix = None
        self.overlap_matrix = None
        self.classify_scores = None
        self.distance_value = None
        self.common_value = None
        self.error_value = None
        self.original_matrix = None
        self.name_list = None
        self.num_list = None
        self.threshold = None
        self.last_classifier = None

    def fit(self, tr_data, tr_labels, ts_data, ts_labels):
        self.class_num = tr_labels.shape[0]
        self.threshold = np.ceil(tr_labels.shape[1] * 0.01)
        unique_matrix, tr_samples, tr_nums = self.sort_pl(tr_labels, tr_data)
        self.coding_matrix, train_pos, train_neg = self.create_matrix(unique_matrix, tr_samples, tr_nums, tr_data, tr_labels)
        self.codingLength = self.coding_matrix.shape[1]
        self.models = self.create_base_models(train_pos, train_neg, ts_data, ts_labels)
        predict_lables, predict_vals = self.get_predict_labels(tr_data, tr_labels)
        # self.reformulate_code_matrix(predict_vals, tr_labels)
        self.performance_matrix = self.create_performance_matrix(predict_lables, tr_labels)
        # self.create_confusion_matrix(tr_data, tr_labels)

    def create_matrix(self, unique_matrix, tr_samples, tr_nums, tr_data, tr_label):
        max_column = 10 * np.log2(self.class_num)
        selected_index = []
        train_pos = []
        train_neg = []
        overlap = 0
        tmp_matrix = copy.deepcopy(unique_matrix)
        matrix = None
        step = 1
        overlap_matrix = np.zeros((unique_matrix.shape[1], unique_matrix.shape[1]))
        while (matrix is None or matrix.shape[1] < max_column) and step <= 2:
            for i1 in range(tmp_matrix.shape[1]):
                if i1 in selected_index:
                    continue
                column_i1 = tmp_matrix[:, i1]
                if tr_nums[i1] < self.threshold:  # 训练样本数要达标
                    continue
                for i2 in range(i1+1, tmp_matrix.shape[1]):
                    if i2 in selected_index:
                        continue
                    column_i2 = tmp_matrix[:, i2]
                    if tr_nums[i2] < self.threshold or np.all(column_i1 * column_i2 == column_i1) \
                        or np.all(column_i1 * column_i2 == column_i2) or sum(column_i1 * column_i2)> overlap:
                        overlap_matrix[i1, i2] = overlap_matrix[i2, i1] = sum(column_i1 * column_i2)
                        continue
                    column = column_i1 + (column_i2 * -1)
                    matrix = column if matrix is None else np.vstack((matrix, column))
                    selected_index.append(i1)
                    selected_index.append(i2)
                    train_pos.append(tr_samples[i1])
                    train_neg.append(tr_samples[i2])
                    break
            step = step + 1
            if step == 2:
                overlap = overlap + 1

        #test
        pos_column = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        pos_ins = tr_data[np.sum(tr_label * np.transpose(np.tile(pos_column, (tr_label.shape[1], 1))), axis=0) == np.sum(tr_label, axis=0), :]
        test_column = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0])
        matrix = np.vstack((matrix, test_column))
        # pos_ins = tr_data[tr_label[0, :] == 1, :]
        train_pos.append(pos_ins)
        neg_ins = tr_data[tr_label[11, :] == 1, :]
        train_neg.append(neg_ins)

        # pos_column = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        # pos_label = np.int8(np.sum(tr_label[:, np.sum(tr_label * np.transpose(np.tile(pos_column, (tr_label.shape[1], 1))), axis=0) == np.sum(tr_label, axis=0)], axis=1) > 0)
        # pos_ins = tr_data[np.sum(tr_label * np.transpose(np.tile(pos_column, (tr_label.shape[1], 1))), axis=0) == np.sum(tr_label, axis=0), :]
        # train_pos.append(pos_ins)
        # neg_colum = np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        # neg_label = -np.int8(np.sum(tr_label[:, np.sum(tr_label * np.transpose(np.tile(neg_colum, (tr_label.shape[1], 1))), axis=0) == np.sum(tr_label, axis=0)], axis=1) > 0)
        # neg_ins = tr_data[np.sum(tr_label * np.transpose(np.tile(neg_colum, (tr_label.shape[1], 1))), axis=0) == np.sum(tr_label, axis=0), :]
        # train_neg.append(neg_ins)
        # test_column = pos_label + neg_label
        # matrix = np.vstack((matrix, test_column))


        return matrix.T, train_pos, train_neg

    def predict(self, ts_data, ts_labels):
        test_label_vector = np.ones(ts_data.shape[0])
        p_test_labels, _, _ = svm_predict(test_label_vector, ts_data.tolist(), self.last_classifier)
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(test_label_vector, ts_data.tolist(), model)
            bin_pre = p_labels if bin_pre is None else np.vstack((bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack((decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.class_num, ts_data.shape[0]))
        # common_value = np.zeros((self.class_num, ts_data.shape[0]))
        # error_value = np.zeros((self.class_num, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.class_num):
                code = self.coding_matrix[j, :]
                # LLW
                # common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] * np.abs(decision_pre_tmp)
                # error = np.int8(bin_pre_tmp == -code) * self.performance_matrix[j, :] * np.abs(decision_pre_tmp)
                # output_value[j, i] = sum(common) - sum(error)
                # ELW
                common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(bin_pre_tmp == -code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common) - sum(error)
                # output_value[j, i] = -sum(self.performance_matrix[j, :] * np.exp(code * decision_pre_tmp))

                # common_value[j, i] = sum(common)
                # error_value[j, i] = sum(error)
                # output_value[j, i] = -sum(distance)
        self.distance_value = -1 * output_value
        # self.common_value = common_value
        # self.error_value = error_value
        pre_label_matrix = np.zeros((self.class_num, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        accuracy = count / ts_data.shape[0]
        return pre_label_matrix, accuracy

    def get_predict_labels(self, tr_data, tr_labels):
        scores = []
        predict_lables = []
        predict_vals = []
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(tr_data.shape[0])
            p_labels, _, p_vals = svm_predict(test_label_vector, tr_data.tolist(), model)
            p_labels = [int(i) for i in p_labels]
            predict_lables.append(p_labels)
            predict_vals.append(p_vals)
            label_vector = np.dot(self.coding_matrix[:, i], tr_labels)
            ts_pos_label = np.array(p_labels)[label_vector > 0]
            ts_neg_label = np.array(p_labels)[label_vector < 0]
            ts_label = np.hstack((ts_pos_label, ts_neg_label))
            tr_pos_label = np.ones(len(label_vector[label_vector > 0]))
            tr_neg_label = -np.ones(len(label_vector[label_vector < 0]))
            tr_label = np.hstack((tr_pos_label, tr_neg_label))
            score = accuracy_score(tr_label, ts_label)
            scores.append(score)

        # 尝试删除性能差的列
        # not_fit_indexes = np.nonzero(np.array(scores) < 0.6)[0]
        # self.coding_matrix = np.delete(self.coding_matrix, not_fit_indexes, axis=1)
        # self.codingLength = self.codingLength - len(not_fit_indexes)
        # self.models = [self.models[i] for i in range(len(self.models)) if i not in not_fit_indexes]
        # scores = [scores[i] for i in range(len(scores)) if i not in not_fit_indexes]
        self.classify_scores = scores
        return predict_lables, predict_vals

    def create_performance_matrix(self, predict_vals, tr_labels):
        performance_matrix = np.zeros((self.class_num, self.codingLength))
        # predict_matrix = None
        for i in range(self.codingLength):
            p_labels = predict_vals[i]
            # predict_matrix = np.array(p_labels) if predict_matrix is None else np.vstack((predict_matrix, np.array(p_labels)))
            for j in range(self.class_num):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                # label_value_j = np.array(p_vals)[tr_labels[j, :] == 1]
                if self.coding_matrix[j, i] == 0:
                    performance_matrix[j, i] = 0
                else:
                    # performance_matrix[j, i] = sum(np.abs(label_value_j[label_class_j == self.coding_matrix[j, i]]))/label_class_j.shape[0]
                    performance_matrix[j, i] = sum(np.int8(label_class_j == self.coding_matrix[j, i]))/label_class_j.shape[0]
            # 计算每个分类器分类正确率
            # label_vector = np.dot(self.coding_matrix[:, i], tr_labels)
            # ts_pos_label = np.array(p_labels)[label_vector > 0]
            # ts_neg_label = np.array(p_labels)[label_vector < 0]
            # ts_label = np.hstack((ts_pos_label, ts_neg_label))
            # tr_pos_label = np.ones(len(label_vector[label_vector > 0]))
            # tr_neg_label = -np.ones(len(label_vector[label_vector < 0]))
            # tr_label = np.hstack((tr_pos_label, tr_neg_label))
            # score = accuracy_score(tr_label, ts_label)
            # scores.append(score)
        performance_matrix = performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

        return performance_matrix

    def create_base_models(self, tr_pos_idx, tr_neg_idx, ts_data, ts_labels):
        models = []
        for i in range(self.codingLength):
            pos_inst = tr_pos_idx[i]
            neg_inst = tr_neg_idx[i]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack((np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            print('第' + str(i) + '个classifier 训练样本数:' + str(len(tr_labels)) + ' 正例：' + str(len(pos_inst)) + ' 负例：' + str(len(neg_inst)))
            # model = self.estimator().fit(tr_inst, tr_labels)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(), tr_inst.tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            _, p_acc, _ = svm_predict(tr_labels.tolist(), tr_inst.tolist(), model)
            print("acc = " + str(p_acc))
            # _, acc, _ = svm_predict(tr_labels.tolist(), tr_inst.tolist(), model)
            # print('正确率：' + str(acc[0]))

            self.test_performance_classify(ts_data, ts_labels, model, self.coding_matrix[:, i])
            models.append(model)
            # test
            if i == 27:
                self.last_classifier = model
        return models

    def test_performance_classify(self, ts_data, ts_labels, model, column):
        pos_inst = ts_data[np.dot(column.T, ts_labels) == 1, :]
        neg_inst = ts_data[np.dot(-column.T, ts_labels) == 1, :]
        inst = np.vstack((pos_inst, neg_inst))
        ins_labels = np.hstack((np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
        p_labels, acc, _ = svm_predict(ins_labels.tolist(), inst.tolist(), model)
        print(acc)

    def create_confusion_matrix(self, tr_datas, tr_labels):
        pre_label_matrix, accuracy = self.predict(tr_datas, tr_labels)
        error_pre_idx = np.sum(tr_labels * pre_label_matrix, axis=0) == 0
        print(pre_label_matrix)

    def reformulate_code_matrix(self, predict_vals, tr_labels):
        model_num = len(self.models)
        scores = np.zeros((self.coding_matrix.shape[0], self.coding_matrix.shape[1]))
        for i in range(model_num):
            pre_vals = predict_vals[i]
            d_count = collections.defaultdict()
            d_num = np.zeros(self.class_num)
            for j in range(tr_labels.shape[1]):
                val = pre_vals[j]
                column = tr_labels[:, j]
                pl_num = sum(column)
                key = tuple(column)
                if key in d_count:
                    d_count[key] = d_count[key] + val/pl_num
                else:
                    d_count[key] = val/pl_num
                d_num = d_num + column
            for m in range(len(d_count)):
                t = d_count.popitem()
                scores[:, i] = scores[:, i] + np.array(t[0]) * t[1]
        scores[scores > 0] = 1
        scores[scores < 0] = -1
        self.original_matrix = self.coding_matrix
        self.coding_matrix = self.coding_matrix * 2 + scores
        self.coding_matrix = self.coding_matrix / np.abs(self.coding_matrix)

    def sigmoid(self, gamma):
        if gamma < 0:
            return 1 - 1/(1 + math.exp(gamma))
        else:
            return 1/(1 + math.exp(-gamma))

    def sort_pl(self, tr_labels, tr_data):

        unique_matrix = self.unique_rows(tr_labels.T).T
        # 每一列的偏标签数
        pl_nums = np.sum(tr_labels, axis=0)
        tr_nums = []
        tr_samples = []
        for i in range(unique_matrix.shape[1]):
            column = unique_matrix[:, i]
            repeat_column = np.transpose(np.tile(column.T, (tr_labels.shape[1], 1)))
            # 训练集，包含偏标签子集
            tr_sample = tr_data[pl_nums == np.sum(tr_labels * repeat_column, axis=0), :]
            tr_num = tr_sample.shape[0]
            tr_nums.append(tr_num)
            tr_samples.append(tr_sample)
        sort_indexs = np.argsort(-np.array(tr_nums))
        tr_nums = [tr_nums[i] for i in sort_indexs]
        tr_samples = [tr_samples[i] for i in sort_indexs]
        unique_matrix = unique_matrix[:, sort_indexs]
        return unique_matrix, tr_samples, tr_nums

    def create_overlap_matrix(self, d_count):
        index_dict = {i: d_count[i] for i in range(len(d_count))}
        matrix = -np.ones((len(d_count, len(d_count))))
        for i in range(len(d_count)):
            pl_i = d_count[i][0]
            for j in range(i+1, len(d_count)):
                pl_j = d_count[j][0]
                overlap = sum(np.array(pl_i) * np.array(pl_j))
                matrix[i, j] = overlap
        return index_dict, matrix

    def unique_rows(self, a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))