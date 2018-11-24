# -*- coding: utf-8 -*-
import os
import time
import copy
import random
import pickle
import graphviz
import networkx as nx
from sklearn import tree, svm
from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from ica_process import over_sampling, SMOTE_over_sampling

trainset_size = 0.8
print 'training set size', trainset_size
run_times = 1
iterations = 20
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', 'neu_reviews', 'pos_reviews', 'neg_reviews']
# attributes_name = ['reviewerID',]


#  divide data by training set and test set
def split_tarinset_testset(graph, attributes):
    print
    print "split train set and test set "
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        # print graph.node[node]
        for attr_name, val in graph.node[node].items():

            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':  # fake means the label in the dataset
                Y_list.append(val)

        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size, random_state=shuffle_stat)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=test_size)

    return X_train, X_test, Y_train, Y_test


#  remove the test set label on the graph data
def remove_test_label(graph, delete_list):
    print
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
# graph_path = "graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle"
graph_path = 'graph/friendship_clean_attitude.pickle'
graph = nx.read_gpickle(graph_path)
macro_sum = 0
micro_sum = 0
recall_sum = 0

for round_num in range(run_times):
    # split
    X_train, X_test, Y_train, Y_test = split_tarinset_testset(graph, attributes_name)

    # remove label of nodes in test set
    current_graph = remove_test_label(graph, X_test)

    ###########################
    zero_zero_num = 0
    for l in X_train:
        node_id = l[0]

        number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
        if current_graph.node[node_id].get('spammer_neighbors_num'):
            # continue  # when use sampling method, use following
            print "some thing wrong with relational attributes", current_graph.node[node_id]

        if number_of_non_spammers == 0 and number_of_spammers == 0:
            zero_zero_num += 1

        current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
        current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
        l.append(number_of_spammers)
        l.append(number_of_non_spammers)
    print 'node 0-0:', zero_zero_num

    for l in X_test:
        node_id = l[0]

        number_of_spammers, number_of_non_spammers = compute_attribute(graph, node_id)
        if graph.node[node_id].get('spammer_neighbors_num'):
            # continue  # when use sampling method, use following
            print "some thing wrong with relational attributes", graph.node[node_id]

        if number_of_non_spammers == 0 and number_of_spammers == 0:
            zero_zero_num += 1
        graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
        graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
        l.append(number_of_spammers)
        l.append(number_of_non_spammers)

    #############################################3
    print X_train[0:5]

    X_train_without_id = [node[1:] for node in X_train]

    print
    print "training classifier"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # over sampling
    # X_train_without_id, Y_train = SMOTE_over_sampling(X_train_without_id, Y_train)

    classifier = tree.DecisionTreeClassifier(criterion="entropy", random_state=shuffle_stat)
    # classifier = svm.SVC()
    classifier.fit(X_train_without_id, Y_train)

    # import pickle
    # print
    # print "load classifier"
    # model_path = 'decision_tree_result/decision_tree_model42.sav'
    # classifier = pickle.load(open(model_path, 'rb'))

    print
    print "complete training"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    Y_predict = list()
    for X_single in X_test:
        X = copy.deepcopy(X_single)
        node_id = X[0]
        label_predict = classifier.predict([X[1:]])
        Y_predict.append(int(label_predict))

    # print current_graph.node[node_id]
    micro = f1_score(Y_test, Y_predict, average='micro')
    macro = f1_score(Y_test, Y_predict, average='macro')
    recall_rate = recall_score(Y_test, Y_predict, average='binary')

    print
    print "f1 macro is", macro
    print "f1 micro is", micro
    print "recall rate is", recall_rate
    recall_sum += recall_rate
    macro_sum += macro
    micro_sum += micro

print
print 'average_recall:', recall_sum/run_times
print 'average_macro:', macro_sum/run_times
print 'average_micro:', micro_sum/run_times
