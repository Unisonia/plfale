import numpy as np
import copy
from Base import Base
from sklearn.metrics import f1_score
import pandas as pd


class PLSVM(Base):

    def __init__(self, params):
        Base.__init__(self, params)
        self.model = None
        self.label_num = None
        self.w = None

    def fit(self, X, Y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """

        # 论文中λ取值范围{0.001，0.01，0.1，1，10，100，1000}，先尝试使用100,正则化参数
        λ = 100
        # 循环周期
        T = 10
        c_len, r_len = X.shape
        self.label_num = Y.shape[0]
        partial_num = np.sum(Y, axis=0)
        # 初始化
        XX = np.repeat(X, self.label_num, axis=0)
        w = 1/(10 * r_len * np.sqrt(λ)) * np.ones(self.label_num * r_len)
        y_ravel = np.ravel(Y, order='F')
        Y_mask = (Y - 1).T
        Y_else_mask = (Y * -1).T
        # θs = np.asarray(θs)
        for t in range(1, T):
            ww = np.tile(w.reshape((self.label_num, r_len)), (c_len, 1))
            w_X = np.sum(np.multiply(XX, ww), axis=1).reshape((c_len, self.label_num))
            S = np.add(w_X, Y_mask)
            S_else = np.add(w_X, Y_else_mask)
            max_S = np.max(S, axis=1)
            argmax_S = np.argmax(S, axis=1)
            max_S_else = np.max(S_else, axis=1)
            argmax_S_else = np.argmax(S_else, axis=1)
            ξ = 1 - np.subtract(max_S, max_S_else)
            A_idx = ξ > 0
            A = X[A_idx, :]
            A_S = argmax_S[A_idx]
            A_S_else = argmax_S_else[A_idx]
            gd = []
            for i in range(self.label_num):
                S_idx = A_S == i
                S_else_idx = A_S_else == i
                tmp = np.zeros(r_len)
                if sum(S_idx) > 0:
                    tmp = tmp + np.sum(X[S_idx], axis=0)
                if sum(S_else_idx) > 0:
                    tmp = tmp - np.sum(X[S_else_idx], axis=0)
                gd.append(tmp)
            gd = np.asarray(gd).reshape((r_len * self.label_num))
            # for i in range(self.label_num):
            #     w_θ = np.dot(θs[i], w)
            #     w_θs.append(w_θ)
            #     w_θ_else = np.dot(θs_else[i], w)
            #     w_θs_else.append(w_θ_else)
            # max_w_θ = np.amax(w_θs, axis=0)
            # max_w_θ_idx = np.argmax(w_θs, axis=0)
            # max_w_θ_else = np.amax(w_θs_else, axis=0)
            # max_w_θ_else_idx = np.argmax(w_θs_else, axis=0)
            # ξ = np.subtract(1, np.subtract(max_w_θ, max_w_θ_else))
            # max_w_θ_idx[ξ <= 0] = -1
            # max_w_θ_else_idx[ξ <= 0] = -1
            # gd = None
            # for i in range(self.label_num):
            #     tmp_max = np.sum(θs[i][max_w_θ_idx == i, :], axis=0)
            #     tmp_min = np.sum(θs_else[i][max_w_θ_else_idx == i, :], axis=0)
            #     tmp = np.subtract(tmp_max, tmp_min)
            #     if gd is None:
            #         gd = tmp
            #     else:
            #         gd = np.add(gd, tmp)
            # # gd = np.sum(np.add(max_θ, min_θ)[ξ > 0], axis=0)
            # # 设置学习率
            η = 1/(λ*t)
            # # 更新w
            w_post = (1 - 1/t) * w + (η/c_len) * gd
            w = min(1, 1/np.sqrt(λ)/np.linalg.norm(w_post, ord=2)) * w_post
        self.w = np.reshape(w, (self.label_num, r_len))

    def predict(self, X):
        c_len, r_len = X.shape
        label = []
        for i in range(c_len):
            x = X[i, :]
            max_index = np.argmax(np.dot(self.w, x))
            x_pred = np.zeros(self.label_num)
            x_pred[max_index] = 1
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
