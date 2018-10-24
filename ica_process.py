# -*- coding: utf-8 -*-
import time
import copy
import pickle
import networkx as nx
from collections import Counter
from sklearn import tree
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, precision_recall_curve, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# configure
run_times = 1
trainset_size = 0.8
print 'training set size', trainset_size
iterations = 10
shuffle_stat = 42
# attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', 'degree']
attributes_name = ['reviewerID','friends_num', 'reviews_num', 'photo_num', ]

# attributes_name = ['reviewerID', ]


def split_trainset_testset(graph, attributes):
    """
    divide data by training set and test set
    :param graph:
    :param attributes:
    :return:
    """
    print "split train set and test set "
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        for attr_name, val in graph.node[node].items():

            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':
                Y_list.append(val)
            elif attr_name == 'degree':
                degree = graph.degree(node)
                Y_list.append(int(degree))

        X_list.append(temp_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size, random_state=shuffle_stat)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size)

    return X_train, X_test, Y_train, Y_test


def remove_test_label(graph, delete_list):
    """
    remove the test set label on the graph data
    :param graph:
    :param delete_list:
    :return:
    """
    print "remove test set label"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    current_graph = graph.copy()
    for node in delete_list:
        current_graph.node[node[0]]['fake'] = 'unknown'
        # print 'remove label of %s' % node[0]
    return current_graph


def compute_attribute(current_graph, node):
    """
    compute attributes according to current network
    :param current_graph:
    :param node:
    :return:
    """
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


def Most_Common(lst):
    """
    return the most common element in the list
    :param lst:
    :return:
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]

# start
graph_path = "graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle"
graph = nx.read_gpickle(graph_path)
macro_sum = 0
micro_sum = 0
recall_sum = 0


for round in range(run_times):
    # split
    X_train, X_test, Y_train, Y_test = split_trainset_testset(graph, attributes_name)
    print 'training set', len(X_train)
    print 'test set', len(X_test)
    final_X_test = copy.deepcopy(X_test)
    final_Y_test = copy.deepcopy(Y_test)

    # remove label of nodes in test set
    current_graph = remove_test_label(graph, X_test)

    # compute relational attributes using only known nodes
    print "computing the relational attributes"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    zero_zero_num = 0
    for l in X_train:
        node_id = l[0]
        number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
        if current_graph.node[node_id].get('spammer_neighbors_num'):
            print "some thing wrong with relational attributes", current_graph.node[node_id]
        if number_of_non_spammers == 0 and number_of_spammers == 0:
            zero_zero_num += 1

        current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
        current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
        l.append(number_of_spammers)
        l.append(number_of_non_spammers)
    print 'node 0-0:', zero_zero_num

    # train
    X_train_without_id = [node[1:] for node in X_train]

    print "training classifier"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    classifier = tree.DecisionTreeClassifier(criterion="entropy")
    classifier.fit(X_train_without_id, Y_train)

    print "complete training"
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    filename = 'decision_tree_result/decision_tree_model%d.sav' % shuffle_stat
    pickle.dump(classifier, open(filename, 'wb'))

    # predict the label of unknown node
    Y_all_predict = defaultdict(list)  # include all the label prediction of each iteration

    specific_node = X_test[0][0]

    for iteration in range(iterations):
        print
        print 'iteration', iteration
        Y_predict = list()

        # generate random order
        X_test, Y_test = shuffle(X_test, Y_test, random_state=iteration)

        for X_single in X_test:
            X = copy.deepcopy(X_single)
            node_id = X[0]

            # compute attributes
            number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
            X.append(number_of_spammers)
            X.append(number_of_non_spammers)
            if number_of_spammers == 0 and number_of_non_spammers == 0 and iteration > 0:
                print 'some thing wrong'
            label_predict = classifier.predict([X[1:]])
            Y_predict.append(int(label_predict))
            Y_all_predict[X[0]].append(int(label_predict))

            # update
            current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
            current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
            current_graph.node[node_id]['fake'] = int(label_predict)

            # if node_id == specific_node:
            #     print current_graph.node[node_id]
            #     neighbors = current_graph.neighbors(node_id)
            #     for neighbor in neighbors:
            #         print 'neighbor', current_graph.node[neighbor]

        micro = f1_score(Y_test, Y_predict, average='micro')
        macro = f1_score(Y_test, Y_predict, average='macro')
        recall_rate = recall_score(Y_test, Y_predict,)
        print
        print "recall rate is ", recall_rate
        print "f1 macro is", macro
        print "f1 micro is", micro
    recall_sum += recall_rate
    macro_sum += macro
    micro_sum += micro

        # calculate the frequency and final label
        # final_label = list()
        # for X in final_X_test:
        #     label_list = Y_all_predict[X[0]]
        #     final_label.append(Most_Common(label_list))
        #
        # print 'final recall rate is', recall_score(final_Y_test, final_label, average='binary')
        # print 'final f1 macro is', f1_score(final_Y_test, final_label, average='macro')
        # print 'final f1 micro is', f1_score(final_Y_test, final_label, average='micro')
        # print

print 'average_recall:', recall_sum/run_times
print 'average_macro:', macro_sum/run_times
print 'average_micro:', micro_sum/run_times


