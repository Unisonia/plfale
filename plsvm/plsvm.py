import numpy as np
from Base import Base
from sklearn.metrics import f1_score


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
        # 初始化w
        w = 1/(10 * r_len * np.sqrt(λ)) * np.ones(r_len * self.label_num)
        for t in range(1, T):
            A = []
            gd = None
            for j in range(c_len):
                x = X[j, :]
                S = Y[:, j]
                if sum(S) == self.label_num:
                    continue
                xx = np.tile(x, (self.label_num, 1))  # 复制x变成矩阵
                y_partial = []
                y_partial_theta = []
                y_else = []
                y_else_theta = []
                for n in range(S.shape[0]):
                    tmp = np.zeros(self.label_num)
                    tmp[n] = 1
                    theta_n = np.ravel(xx * np.tile(tmp, (r_len, 1)).T)
                    w_theta_n = np.dot(w, theta_n)
                    if S[n] != 0:
                        y_partial.append(w_theta_n)
                        y_partial_theta.append(theta_n)
                    else:
                        y_else.append(w_theta_n)
                        y_else_theta.append(theta_n)

                if 1 - (max(y_partial) - max(y_else)) > 0:
                    A.append(x)
                    δ = y_partial_theta[np.argmax(y_partial)] - y_else_theta[np.argmax(y_else)]
                    if gd is None:
                        gd = δ
                    else:
                        gd = gd + δ
            # 设置学习率
            η = 1/(λ*t)
            # 更新w
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
