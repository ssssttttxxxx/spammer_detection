# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import decimal
from sklearn.model_selection import train_test_split


def simple_test():
    X, y = np.arange(10).reshape((5, 2)), range(5)
    print X, y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print X_train
    print y_train
    print X_test
    print y_test


def split_trainset_testset(graph, attributes, training_set_size):
    """
    divide data by training set and test set
    :param graph:
    :param attributes:
    :return:
    """
    # print
    # print "split train set and test set "
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    random_stat = 42

    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()

        for attr_name, val in graph.node[node].items():
            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':
                Y_list.append(val)

        if 'degree' in attributes:
            degree = graph.degree(node)
            temp_list.append(int(degree))

        X_list.append(temp_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=float(1-training_set_size),
                                                        random_state=random_stat)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":

    attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', 'degree', 'pos_reviews', 'neg_reviews',
                       'neu_reviews', 'betweenness', 'closeness']  # degree variable is useless

    Graph_path = '../graph/old_friendship_connected_closeness_centrality.pkl'
    Graph = nx.read_gpickle(Graph_path)
    # training_set_size = 0.1

    training_set_list = [decimal.Decimal(x) / decimal.Decimal(10) for x in range(1, 10)]
    for training_set_size in training_set_list:
        x_train, x_test, y_train, y_test = split_trainset_testset(Graph, attributes_name, training_set_size)

        print type(x_train)
        print len(x_train)
        print x_train[1]

        x_train_np = np.array(x_train)
        x_test_np = np.array(x_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        np.savetxt('../train_test_data_set_ass1/%s/training_set.csv' % training_set_size, x_train_np, delimiter=',', encoding='utf8', fmt='%s')
        np.savetxt('../train_test_data_set_ass1/%s/test_set.csv' % training_set_size, x_test_np, delimiter=',', encoding='utf8', fmt='%s')
        np.savetxt('../train_test_data_set_ass1/%s/label_training_set.csv' % training_set_size, y_train_np, delimiter=',', encoding='utf8', fmt='%s')
        np.savetxt('../train_test_data_set_ass1/%s/label_test_set.csv' % training_set_size, y_test_np, delimiter=',', encoding='utf8', fmt='%s')
