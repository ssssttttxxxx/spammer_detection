# -*- coding: utf-8 -*-
import time
import copy
import pickle
import networkx as nx
from sklearn import tree
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split

# configure
run_times = 10
trainset_size = 0.5
iterations = 10
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num']
# attributes_name = ['reviewerID', ]


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

        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size, random_state=shuffle_stat)
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
macro_sum = 0
micro_sum = 0
recall_sum = 0

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

print "training classifier"
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train_without_id, Y_train)

print "complete training"
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# predict the label of unkonwn node
Y_all_predict = list()  # include all the label prediction of each iteration
next_graph = current_graph.copy()

for iteration in range(iterations):
    print 'iteration', iteration
    Y_predict = list()
    X_test, Y_test = shuffle(X_test, Y_test, random_state=iteration)

    if iteration > 0:
        current_graph = next_graph.copy()
    # random.shuffle(X_test)
    for X_single in X_test:
        X = copy.deepcopy(X_single)
        node_id = X[0]

        # compute attributes
        number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
        X.append(number_of_spammers)
        X.append(number_of_non_spammers)

        label_predict = classifier.predict([X[1:]])
        Y_predict.append(label_predict)

        # update
        next_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
        next_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
        next_graph.node[node_id]['fake'] = label_predict

    micro = f1_score(Y_test, Y_predict, average='micro')
    macro = f1_score(Y_test, Y_predict, average='macro')
    recall_rate = recall_score(Y_test, Y_predict, average='binary')
    print "f1 macro is", macro
    print "f1 micro is", micro
    print "recall rate is", recall_rate
    recall_sum += recall_rate
    macro_sum += macro
    micro_sum += micro

# print 'average_recall:', recall_sum / run_times
# print 'average_macro:', macro_sum / run_times
# print 'average_micro:', micro_sum / run_times

