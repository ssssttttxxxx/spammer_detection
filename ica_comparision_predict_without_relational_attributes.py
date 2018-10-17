# -*- coding: utf-8 -*-
import os
import time
import copy
import random
import pickle
import graphviz
import networkx as nx
from sklearn import tree
from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

test_size = 0.5
run_times = 5
iterations = 20
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photos_num']
# attributes_name = ['reviewerID',]



#  divide data by training set and test set
def split_tarinset_testset(graph, attributes):

    print "split train set and test set "
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        for attr_name, val in graph.node[node].items():

            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':  # fake means the label in the dataset
                Y_list.append(val)

        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=0.2, random_state=shuffle_stat)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=test_size)

    return X_train, X_test, Y_train, Y_test


#  remove the test set label on the graph data
def remove_test_label(graph, delete_list):
    print "remove test set label"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    current_graph = graph.copy()
    for node in delete_list:
        current_graph.node[node[0]]['fake'] = 'unknown'
    return current_graph


# compute attributes according to current network
def compute_attribute(current_graph, node):
    neighbors = current_graph.neighbors(node)
    number_of_spammers = 0
    number_of_non_spammers = 0

    for neighbor in neighbors:
        if current_graph.node[neighbor]['fake'] == 1:
            number_of_spammers += 1
        elif current_graph.node[neighbor]['fake'] == 0:
            number_of_non_spammers += 1
        else:
            continue
    return number_of_spammers, number_of_non_spammers


# start
graph_path = "graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle"
graph = nx.read_gpickle(graph_path)
macro_sum = 0
micro_sum = 0
recall_sum = 0

for k in range(run_times):
    # split
    X_train, X_test, Y_train, Y_test = split_tarinset_testset(graph, attributes_name)

    # check attributes
    # print X_train[0]
    # print graph.node[X_train[0][0]]

    # remove label of nodes in test set
    current_graph = remove_test_label(graph, X_test)

    X_train_without_id = [node[1:] for node in X_train]

    print "training classifier"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    classifier = tree.DecisionTreeClassifier(criterion="entropy")
    classifier.fit(X_train_without_id, Y_train)

    print "complete training"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    next_graph = current_graph.copy()

    Y_predict = list()
    for X_single in X_test:
        X = copy.deepcopy(X_single)
        node_id = X[0]

        try:
            label_predict = classifier.predict([X[1:]])
        except:
            print X[1:]
            print current_graph.node[node_id]
            exit(1)
        Y_predict.append(int(label_predict))

    # print current_graph.node[node_id]
    print "iAccuracy is", accuracy_score(Y_test, Y_predict) * 100
    micro = f1_score(Y_test, Y_predict, average='micro')
    macro = f1_score(Y_test, Y_predict, average='macro')
    recall_rate = recall_score(Y_test, Y_predict, average='binary') * 100
    print "f1 macro is", macro
    print "f1 micro is", micro
    print "recall rate is", recall_rate
    print "iPrecision is ", precision_score(Y_test, Y_predict) * 100
    recall_sum += recall_rate
    macro_sum += macro
    micro_sum += micro

print 'average_recall:', recall_sum/run_times
print 'average_macro:', macro_sum/run_times
print 'average_micro:', micro_sum/run_times
