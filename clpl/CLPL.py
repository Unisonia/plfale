import numpy as np
from Base import Base
from liblinearutil import *
from sklearn.metrics import f1_score


class CLPL(Base):
    def __init__(self, params):
        Base.__init__(self, params)
        # self._kernel = kernel
        # self._c = c
        self.model = None
        self.label_num = None

    def fit(self, X, Y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """

        # 获取训练数据
        tr_negatives = None
        # c_len, r_len = X.shape
        self.label_num = Y.shape[0]
        repeat_x = np.repeat(X, self.label_num, axis=0)
        repeat_y = np.tile(np.ravel(Y, order='F'), (X.shape[1], 1)).T
        tr_positives = repeat_x * repeat_y
        tr_positives = tr_positives.reshape((X.shape[0], X.shape[1] * self.label_num))
        for i in range(Y.shape[0]):
            neg = X[Y[i, :] == 0, :]
            repeat_neg = np.repeat(neg, self.label_num, axis=0)
            neg_label = np.zeros(self.label_num)
            neg_label[i] = 1
            repeat_neg_label = np.tile(neg_label.T, (X.shape[1], 1)).T
            repeat_neg_label = np.tile(repeat_neg_label, (neg.shape[0], 1))
            tr_negative = repeat_neg * repeat_neg_label
            tr_negatives = tr_negative if tr_negatives is None else np.vstack((tr_negatives, tr_negative))
        # for i in range(c_len):
        #     x = X[i, :]
        #     S = Y[:, i]
        #     xx = np.tile(x, (self.label_num, 1))  # 复制x变成矩阵
        #     pos = np.ravel(xx * np.tile(S, (r_len, 1)).T)
        #     for j in range(S.shape[0]):
        #         if S[j] != 0:
        #             continue
        #         else:
        #             tmp = np.zeros(self.label_num)
        #             tmp[j] = 1
        #             neg = np.ravel(xx * np.tile(tmp, (r_len, 1)).T)
        #             tr_negatives = neg if tr_negatives is None else np.vstack((tr_negatives, neg))
        #     tr_positives = pos if tr_positives is None else np.vstack((tr_positives, pos))
        assert tr_positives is not None
        assert tr_negatives is not None
        tr_negatives = tr_negatives.reshape((int(tr_negatives.shape[0]/self.label_num), X.shape[1] * self.label_num))
        tr_pos_label = np.ones(tr_positives.shape[0])
        tr_neg_label = -np.ones(tr_negatives.shape[0])
        tr_data = np.vstack((tr_positives, tr_negatives))
        tr_label = np.hstack((tr_pos_label, tr_neg_label))
        prob = problem(tr_label, tr_data)
        param = parameter('-s 2 -c 5 -q')
        self.model = train(prob, param)

    def predict(self, X):
        tmp_label = np.ones(self.label_num)
        c_len, r_len = X.shape
        # 生成数据预处理矩阵 θ
        transform_matrix = np.zeros((self.label_num, r_len*self.label_num))
        for i in range(self.label_num):
            transform_matrix[i, r_len*i:r_len*(i+1)] = 1
        label = []
        for i in range(c_len):
            tmp = np.tile(X[i, :], (self.label_num, self.label_num))
            x = transform_matrix * tmp
            p_labs, _, p_vals = predict(tmp_label, x, self.model, options="-q")
            x_pred = np.zeros(self.label_num)
            x_pred[np.argmax(p_vals)] = 1
            label.append(x_pred)
        return np.asarray(label).T

    def score(self, X, y_test):
        y_predict = self.predict(X)
        result = y_predict * y_test
        labels = list(range(y_test.shape[0]))
        p_label = np.dot(labels, y_predict)
        true_label = np.dot(labels, y_test)
        fscore = f1_score(true_label, p_label, labels, average='macro')
        return y_predict, sum(np.sum(result, axis=0))/len(X), fscore
