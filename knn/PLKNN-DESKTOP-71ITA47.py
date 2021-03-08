import numpy as np
from Base import Base
from sklearn.preprocessing import normalize
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from sklearn.metrics import euclidean_distances


class PLKNN(Base):
    def __init__(self, params):
        Base.__init__(self, params)
        self.k = self.params.get("k", 10)
        assert self.k >= 1, 'k must be valid'
        self._x_train = None
        self._y_train = None
        self._class_num = None

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._class_num = y_train.shape[0]
        self._y_train = y_train
        return self

    # def _predict(self, x):
    #     d = [np.sqrt(np.sum((self._x_train[i, :]-x)**2)) for i in range(self._x_train.shape[0])]
    #     euclidean_distances(np.array(x), np.asarray(self._x_train.tolist()))
    #     nearest = np.argsort(d)
    #     k_total_d = sum(np.array(d)[nearest[:self.k]])
    #     top_k = [(1-d[i]/k_total_d) * self._y_train[:, i] for i in nearest[:self.k]]
    #     votes = np.sum(np.asarray(top_k).T, axis=1)
    #     predict = np.zeros(self._class_num)
    #     predict[np.argmax(votes)] = 1
    #     return predict

    def predict(self, x_predict):
        dist = euclidean_distances(x_predict, self._x_train.tolist())
        nearest = np.argsort(dist, axis=1)
        y_predict = []
        for i in range(nearest.shape[0]):
            k_total_d = sum(dist[i, nearest[i, :self.k]])
            top_k = [(1-dist[i, j]/k_total_d) * self._y_train[:, j] for j in nearest[i, :self.k]]
            votes = np.sum(np.asarray(top_k).T, axis=1)
            predict = np.zeros(self._class_num)
            predict[np.argmax(votes)] = 1
            y_predict.append(predict)
        return np.asarray(y_predict).T

    def __repr__(self):
        return 'knn(k=%d):' % self.k

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        result = y_predict * y_test
        labels = list(range(y_test.shape[0]))
        p_label = np.dot(labels, y_predict)
        true_label = np.dot(labels, y_test)
        fscore = f1_score(true_label, p_label, labels, average='macro')
        return y_predict, sum(np.sum(result, axis=0))/len(x_test), fscore
