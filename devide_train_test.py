# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split


def split_tarinset_testset(graph_path):
    graph = nx.read_gpickle(graph_path)
    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        for attr_name, val in graph.node[node]:
            if attr_name == 'fake':  # fake即为label值
                Y_list.append(val)
            else:
                temp_list.append(val)
        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":

    X, y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print X_train
    print y_train
    print X_test
    print y_test
