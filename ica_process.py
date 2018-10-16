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
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iterations = 10
shuffle_stat = 117
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
            # if attr_name == 'fake':  # fake即为label值
            #     Y_list.append(val)
            # # elif attr_name == 'reviewerName' or attr_name == 'location' or attr_name == 'reviewerID':
            # elif attr_name == 'reviewerName' or attr_name == 'location':
            #     continue
            # else:
            #     temp_list.append(val)
            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':
                Y_list.append(val)


        # print temp_list
        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=0.8, random_state=shuffle_stat)
    # print X_train
    # print y_train
    # print X_test
    # print y_test
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

# split
X_train, X_test, Y_train, Y_test = split_tarinset_testset(graph, attributes_name)

# check attributes
# print X_train[0]
# print graph.node[X_train[0][0]]

# remove label of nodes in test set
current_graph = remove_test_label(graph, X_test)

# compute relational attributes
print "computing the relational attributes"
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
for l in X_train:
    node_id = l[0]
    number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
    if current_graph.node[node_id].get('spammer_neighbors_num'):
        print "??", current_graph.node[node_id]
    current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
    current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
    l.append(number_of_spammers)
    l.append(number_of_non_spammers)


# train
X_train_without_id = [node[1:] for node in X_train]
# check attributes order in list
print X_train_without_id[0]
print X_train[0]
print "training classifier"
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train_without_id, Y_train)

# visualize and store decision tree
# tree_data = tree.export_graphviz(classifier, out_file=None)
# tree_graph = graphviz.Source(tree_data)
# tree_graph.render("decision_tree_result/spammer_decision")

print "complete training"
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

filename = 'decision_tree_result/decision_tree_model%d.sav' % shuffle_stat
pickle.dump(classifier, open(filename, 'wb'))

# predict the label of unkonwn node

Y_all_predict = list()  # include all the label prediction of each iteration


for iteration in range(iterations):
    print 'iteration', iteration
    Y_predict = list()
    random.Random(iteration).shuffle(X_test)
    # random.shuffle(X_test)
    for X_single in X_test:
        X = copy.deepcopy(X_single)
        node_id = X[0]
        # print X
        # print node_id

        # compute attributes
        number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
        X.append(number_of_spammers)
        X.append(number_of_non_spammers)
        # print X[1:]
        try:
            label_predict = classifier.predict([X[1:]])
        except:
            print X[1:]
            print current_graph.node[node_id]
            exit(1)
        Y_predict.append(int(label_predict))

        # update
        current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
        current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
        current_graph.node[node_id]['fake'] = label_predict
        # print current_graph.node[node_id]
    print "iAccuracy is ", accuracy_score(Y_test, Y_predict) * 100

    if iteration > 0:
        if Y_predict == Y_all_predict[-1]:
            print "same"
        else:
            print "not same"
    Y_all_predict.append(Y_predict)
# print Y_predict
# print Y_test


# calculate the frequency and final label
all_node_label = list()
for i in range(len(Y_test)):
    node_label = list()
    for j in range(iterations):
        node_label.append(int(Y_all_predict[j][i]))
    all_node_label.append(node_label)

final_labels = list()
for node_label in all_node_label:
    most_common_label, num_most_common = Counter(node_label).most_common(1)[0]
    final_labels.append(most_common_label)

print "Accuracy is ", accuracy_score(Y_test, final_labels)*100
