# -*- coding: utf-8 -*-
import time
import copy
import pickle
import random
import pandas as pd
from numpy import array
import pickle
import networkx as nx
from collections import Counter
from sklearn import tree
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, precision_recall_curve, recall_score, \
    classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# configure
run_times = 1
training_set_size = 0.9
iterations = 10
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', 'degree', 'pos_reviews', 'neg_reviews', 'neu_reviews']  # degree variable is useless
# attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', ]
# attributes_name = ['reviewerID', ]


def split_trainset_testset(graph, attributes):
    """
    divide data by training set and test set
    :param graph:
    :param attributes:
    :return:
    """
    # print
    # print "split train set and test set "
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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
    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1 - training_set_size,
                                                        random_state=shuffle_stat)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size)

    return X_train, X_test, Y_train, Y_test


def split_trainset_testset_deepwalk(graph, attributes):
    X_list = list()
    Y_list = list()

    with open('embeddings/friendship_reviewer_label_attr_clean_unknown_degree0.embeddings', 'r') as embeddings:
    # with open('embeddings/high_degree.embeddings', 'r') as embeddings:

        summary = embeddings.readline().split()
        num_of_nodes = summary[0].strip()
        num_of_dimension = summary[1].strip()

        print summary

        for line in embeddings:
            features = line.strip().split(' ')
            node_id = features[0]

            for attr_name, val in graph.node[node_id].items():
                if attr_name != 'reviewerID':
                    if attr_name in attributes:
                        features.append(val)
                    elif attr_name == 'fake':
                        Y_list.append(int(val))

            if 'degree' in attributes:
                degree = graph.degree(node_id)
                features.append(int(degree))

            X_list.append(features)

    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1 - training_set_size,
                                                        random_state=shuffle_stat)
    return X_train, X_test, Y_train, Y_test


def remove_test_label(graph, delete_list):
    """
    remove the test set label on the graph data
    :param graph: original graph
    :param delete_list: test set for X
    :return:
    """
    # print
    # print "remove test set label"
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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


def over_sampling(X_train, X_label):
    spammer_list = list()
    non_spammer_list = list()

    sampled_train_list = list()
    sampled_label_list = list()

    for attrs, label in zip(X_train, X_label):
        if label == 1:
            spammer_list.append(attrs)
        elif label == 0:
            non_spammer_list.append(attrs)

    for i in range(len(X_train)):
        if i % 2 == 0:
            sampled_train_list.append(random.sample(spammer_list, 1)[0])
            sampled_label_list.append(int(1))
        else:
            sampled_train_list.append(random.sample(non_spammer_list, 1)[0])
            sampled_label_list.append(int(0))

    return sampled_train_list, sampled_label_list


def SMOTE_over_sampling(X_train, X_label):
    X_train = array(X_train).astype(float)
    X_label = array(X_label).astype(float)
    print('Original dataset shape {}'.format(Counter(X_label)))

    sm = SMOTE(0.8, random_state=shuffle_stat)
    over_samples_X, over_samples_Y = sm.fit_sample(X_train, X_label)
    print("After OverSampling, counts of label '1': {}".format(sum(over_samples_Y == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(over_samples_Y == 0)))
    return over_samples_X, over_samples_Y


def compute_second_degree_attributes(current_graph, node):
    neighbors = current_graph.neighbors(node)
    number_of_second_spammers = 0
    number_of_second_non_spammers = 0
    for neighbor in neighbors:
        second_neighbors = current_graph.neighbors(neighbor)
        for second_neighbor in second_neighbors:
            if current_graph.node[second_neighbor]['fake'] == 1:
                number_of_second_spammers += 1
            elif current_graph.node[second_neighbor]['fake'] == 0:
                number_of_second_non_spammers += 1
            else:
                continue

    return number_of_second_spammers, number_of_second_non_spammers


def aggregation_percentage(label1_num, label2_num):
    sum_num = label1_num + label2_num
    if sum_num == 0:
        return sum_num, sum_num
    label1_percentage = float(label1_num)/sum_num
    label2_percentage = float(label2_num)/sum_num
    # print label1_percentage, label2_percentage
    return label1_percentage, label2_percentage


if __name__ == '__main__':
    # start
    print 'training set size', training_set_size

    graph_path = "graph/firendship_new_label209579.pickle"
    graph = nx.read_gpickle(graph_path)

    for round_num in range(run_times):
        # split into train set and test set
        X_train, X_test, Y_train, Y_test = split_trainset_testset(graph, attributes_name)
        Y_train = array(Y_train).astype(float)
        Y_test = array(Y_test).astype(float)

        print
        print 'training set', len(X_train)
        print 'test set', len(X_test)

        print "training set '1': {}".format(sum(Y_train == 1))
        print "test set '1' : {}".format(sum(Y_test == 1))
        print 'label 1 ratio: ', float(sum(Y_train == 1)) / (sum(Y_train == 1) + sum(Y_test == 1))

        print "training set '0': {}".format(sum(Y_train == 0))
        print "test set '0' : {}".format(sum(Y_test == 0))
        print 'label 0 ratio: ', float(sum(Y_train == 0)) / (sum(Y_train == 0) + sum(Y_test == 0))

        # final_X_test = copy.deepcopy(X_test)
        # final_Y_test = copy.deepcopy(Y_test)

        # remove label of nodes in test set
        current_graph = remove_test_label(graph, X_test)

        # processing the training set
        zero_zero_num = 0  # record the number of node whose neighbors are all belonged to test set

        for l in X_train:
            node_id = l[0]

            number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
            # spammers_percentage, non_spammers_persentage = aggregation_percentage(number_of_spammers, number_of_non_spammers)

            # number_of_second_spammers, number_of_second_non_spammer = compute_second_degree_attributes(current_graph,
            #                                                                                            node_id)

            if current_graph.node[node_id].get('spammer_neighbors_num'):
                # continue  # when use sampling method, use following
                print "some thing wrong with relational attributes", current_graph.node[node_id]

            if number_of_non_spammers == 0 and number_of_spammers == 0:
                zero_zero_num += 1

            current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
            current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
            l.append(number_of_spammers)
            l.append(number_of_non_spammers)
            # l.append(number_of_second_spammers)
            # l.append(number_of_second_non_spammer)

        X_train_without_id = [node[1:] for node in X_train]
        X_train_without_id = array(X_train_without_id).astype(float)
        Y_train = array(Y_train).astype(float)

        # over sampling
        # X_over_sample, Y_over_sample = SMOTE_over_sampling(X_train_without_id, Y_train)

        # train the classifier
        print
        print 'dimension of training features', len(X_train_without_id[0])
        classifier = tree.DecisionTreeClassifier(random_state=shuffle_stat)
        # classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
        # classifier = GaussianNB()
        # classifier = RandomForestClassifier()

        # if don't use over sampling
        # cross validate
        validate_score = cross_val_score(classifier, X_train_without_id, Y_train,
                                         cv=10)
        print 'cross validate score', validate_score
        classifier.fit(X_train_without_id, Y_train)

        # if use over sampling
        # cross validate
        # validate_score = cross_val_score(classifier, X_over_sample, Y_over_sample,
        #                                  cv=10)
        # print 'cross validate score', validate_score
        # classifier.fit(X_over_sample, Y_over_sample)

        # save the classifier in pickle
        # filename = 'decision_tree_result/decision_tree_model%d.sav' % shuffle_stat
        # pickle.dump(classifier, open(filename, 'wb'))

        # load classifier
        # model_path = 'decision_tree_result/decision_tree_model42.sav'
        # classifier = pickle.load(open(model_path, 'rb'))

        # check the prediction result on training set
        Y_train_predict = list()
        for l in X_train:
            Y_train_predict.append(classifier.predict([l[1:]]))
        print classification_report(Y_train, Y_train_predict)
        print '----------------------------------------------------------------'

        # check the performance of classifier without collective classification
        Y_predict = list()
        X_test_copy = copy.deepcopy(X_test)
        for X in X_test_copy:
            node_id = X[0]
            number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
            # spammers_percentage, non_spammers_persentage = aggregation_percentage(number_of_spammers,
            #                                                                       number_of_non_spammers)
            # number_of_second_spammers, number_of_second_non_spammer = compute_second_degree_attributes(current_graph,
            #                                                                                            node_id)

            X.append(number_of_spammers)
            X.append(number_of_non_spammers)
            # X.append(number_of_second_spammers)
            # X.append(number_of_second_non_spammer)
            X_test_without_id = X[1:]
            X_test_fit = array(X_test_without_id).astype(float)
            Y_ = classifier.predict([X_test_fit])
            Y_predict.append(Y_)

            # update
            current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
            current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
            current_graph.node[node_id]['fake'] = int(Y_)

        print classification_report(Y_test, Y_predict, digits=4)
        print '------------------------------------------------------------------'

        # ICA
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
                # spammers_percentage, non_spammers_persentage = aggregation_percentage(number_of_spammers,
                #                                                                       number_of_non_spammers)

                # number_of_second_spammers, number_of_second_non_spammer = compute_second_degree_attributes(current_graph, node_id)
                X.append(number_of_spammers)
                X.append(number_of_non_spammers)
                # X.append(number_of_second_spammers)
                # X.append(number_of_second_non_spammer)
                X_test_without_id = X[1:]
                X_test_fit = array(X_test_without_id).astype(float)
                label_predict = classifier.predict([X_test_fit])
                Y_predict.append(int(label_predict))
                Y_all_predict[X[0]].append(int(label_predict))

                # update
                current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
                current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
                current_graph.node[node_id]['fake'] = int(label_predict)

            micro = f1_score(Y_test, Y_predict, average='micro')
            macro = f1_score(Y_test, Y_predict, average='macro')
            recall_rate = recall_score(Y_test, Y_predict)
            acc = accuracy_score(Y_test, Y_predict)
            print classification_report(Y_test, Y_predict, digits=6)
            # print "recall rate is ", recall_rate
            # print "f1 macro is", macro
            # print "f1 micro is", micro
