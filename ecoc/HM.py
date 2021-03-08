import numpy as np
import collections
from ecoc.BasePLECOC import BasePLECOC
from svmutil import *
from sklearn.metrics import accuracy_score
from tools import Tools
# from sklearn.svm import libsvm


class BottomUpPLECOC(BasePLECOC):

    def __init__(self, params):
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

    def fit(self, tr_data, tr_labels):
        self.class_num = tr_labels.shape[0]
        self.coding_matrix, self.overlap_matrix, tr_pos_idx, tr_neg_idx = self.create_matrix(tr_data, tr_labels)
        self.codingLength = self.coding_matrix.shape[1]
        self.models = self.create_base_models(tr_pos_idx, tr_neg_idx)
        predict_vals = self.get_predict_labels(tr_data, tr_labels)
        self.reformulate_code_matrix(predict_vals, tr_labels)
        self.performance_matrix = self.create_performance_matrix(predict_vals, tr_labels)


    def predict(self, ts_data, ts_labels):
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
                output_value[j, i] = -sum(common)-sum(error)
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

    def create_matrix(self, tr_data, tr_labels):
        self.class_num = tr_labels.shape[0]
        d_count, d_index = self.create_frequency_dict(tr_labels)
        matrix = None
        tr_pos_idx = []
        tr_neg_idx = []
        overlap_matrix = None
        self.threshold = np.ceil(0.1 * tr_data.shape[0])
        overlap_num = 0
        num = 1
        while overlap_num < self.class_num:
            fail_match = []
            while d_count:
                flag = False
                node = d_count.pop(0)
                data_index = d_index.get(node[0])
                for i in range(len(d_count)):
                    nx_node = d_count[i]
                    # column = np.array(node[0]) + (np.array(nx_node[0]) * -1)
                    # 交集数需要<= overlap_num 才进行配对
                    intersection = np.array(node[0]) & np.array(nx_node[0])
                    if sum(intersection) > overlap_num or np.all(intersection == np.array(node[0])) \
                            or np.all(intersection == np.array(nx_node[0])):
                        # if len(column[column > 0]) < 1 and len(column[column < 0]) < 1:
                        continue
                    else:
                        column = np.array(node[0]) + (np.array(nx_node[0]) * -1) + (np.array(node[0]) * np.array(nx_node[0]))
                        flag = True
                        matrix = column if matrix is None else np.vstack((matrix, column))
                        tr_pos_idx.append(tr_data[data_index, :])
                        nx_data_index = d_index.get(nx_node[0])
                        tr_neg_idx.append(tr_data[nx_data_index, :])
                        overlap_column = 1/(np.array(node[0]) * np.array(nx_node[0]) + 1)
                        overlap_matrix = overlap_column if overlap_matrix is None else np.vstack((overlap_matrix, overlap_column))
                        break
                # 未配对成功加入队尾
                if not flag:
                    fail_match.append(node)
                else:
                    d_count.pop(i)
                    key = tuple(np.int8(np.array(node[0]) + np.array(nx_node[0]) > 0))
                    d_count.append(tuple((key, node[1] + nx_node[1])))
                    d_index[key] = data_index + nx_data_index
            num = num + 1
            d_count = fail_match
            overlap_num = overlap_num + 1
        matrix = matrix.T
        overlap_matrix = overlap_matrix.T
        return matrix, overlap_matrix, tr_pos_idx, tr_neg_idx

    def create_frequency_dict(self, tr_labels):
        threshold = int(tr_labels.shape[1] * 0.01)
        d_count = collections.defaultdict()
        d_index = collections.defaultdict()
        for i in range(tr_labels.shape[1]):
            column = tr_labels[:, i]
            t = tuple(column)
            if t in d_count:
                d_count[t] += 1
                d_index[t].append(i)
            else:
                d_count[t] = 1
                d_index[t] = [i]
        # 降序排序
        d_count = sorted(d_count.items(), key=lambda d: d[1], reverse=True)
        # 合并弱小节点，保证每个节点都有1%的样本
        while d_count:
            t1 = d_count.pop()
            if t1[1] < threshold:
                if d_count:
                    t2 = d_count.pop()
                    key = tuple(np.int8(np.array(t1[0]) + np.array(t2[0]) > 0))
                    d_count.append(tuple((key, t1[1] + t2[1])))
                    d_index[key] = d_index[t1[0]] + d_index[t2[0]]
                    d_count = sorted(d_count, key=lambda d: d[1], reverse=True)
            else:
                d_count.append(t1)
                break
        return d_count, d_index

    def get_predict_labels(self, tr_data, tr_labels):
        scores = []
        predict_vals = []
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(tr_data.shape[0])
            p_labels, _, p_vals = svm_predict(test_label_vector, tr_data.tolist(), model)
            p_labels = [int(i) for i in p_labels]
            predict_vals.append(p_labels)
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
        return predict_vals

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

    def create_base_models(self, tr_pos_idx, tr_neg_idx):
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
            # _, p_acc, _ = svm_predict(tr_labels.tolist(), tr_inst.tolist(), model)
            # print("acc = " + str(p_acc))
            _, acc, _ = svm_predict(tr_labels.tolist(), tr_inst.tolist(), model)
            print('正确率：' + str(acc[0]))
            models.append(model)
        return models

    def reformulate_code_matrix(self, predict_vals, tr_labels):
        model_num = len(self.models)
        scores = np.zeros((self.coding_matrix.shape[0], self.coding_matrix.shape[1]))
        for i in range(model_num):
            pre_vals = predict_vals[i]
            for j in range(tr_labels.shape[1]):
                val = pre_vals[j]
                partial_num = sum(tr_labels[:, j])
                scores[tr_labels[:, j] > 0, i] = scores[tr_labels[:, j] > 0, i] + val/partial_num
        scores[scores > 0] = 1
        scores[scores < 0] = -1
        self.original_matrix = self.coding_matrix
        self.coding_matrix = self.coding_matrix * 2 + scores
        self.coding_matrix = self.coding_matrix / np.abs(self.coding_matrix)


