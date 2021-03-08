import pandas as pd
from sklearn.feature_selection import SelectKBest


def read(path):
    df = pd.read_csv(path, header=None)
    df_values = df.values
    columns_num = df_values.shape[1]
    data = df_values[:, 0:columns_num-1]
    label = df_values[:, columns_num-1]
    return data, label


def read_and_fs(path, header=None):
    df = pd.read_csv(path, header=None)
    df_values = df.values
    columns_num = df_values.shape[1]
    data = df_values[:, 0:columns_num-1]
    label = df_values[:, columns_num-1]
    data = SelectKBest(k=100).fit_transform(data, label)
    return data, label