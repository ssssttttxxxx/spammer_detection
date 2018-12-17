# -*- coding: utf-8 -*-
import copy
import decimal
import pickle
import random
from numpy import array
import networkx as nx
from collections import Counter
from sklearn import tree
from sklearn.utils import shuffle
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from read_training_test_set import read_data_from_file
from func_may_useful.write_report_in_file import classification_report_csv

# configure
standardize_flag = 1  # 1 for turning on standardization
run_times = 1
# training_set_size = 0.3
training_set_list = [decimal.Decimal(x)/decimal.Decimal(10) for x in range(1, 10)]
classifier_name = 'Naive Bayes ass3'
print training_set_list
iterations = 20
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num',
                   'reviews_num', 'photo_num',
                   'degree', 'pos_reviews',
                   'neg_reviews', 'neu_reviews',
                   'betweenness', 'closeness']
updating_attr_name = ['spammers_num', 'legitimates_num']
# degree variable is useless
# attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num', ]
# attributes_name = ['reviewerID', ]
dimension_of_attr = len(attributes_name) + len(updating_attr_name) - 1


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
        if i == 1:
            print graph.node[node], 'spammers_num', 'legitimates_num'

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


def load_trainset_testset(training_size):
    tr_path = 'train_test_data_set/%s/training_set.csv' % training_size
    tr_label_path = 'train_test_data_set/%s/label_training_set.csv' % training_size
    te_path = 'train_test_data_set/%s/test_set.csv' % training_size
    te_label_path = 'train_test_data_set/%s/label_test_set.csv' % training_size

    tr, tr_label, te, te_label = read_data_from_file(tr_path, tr_label_path, te_path, te_label_path)
    return tr.tolist(), te.tolist(), tr_label.tolist(), te_label.tolist()


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


def most_common(lst):
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


def random_over_sampling(X_train, X_label):
    X_train = array(X_train).astype(float)
    X_label = array(X_label).astype(float)
    print('Original dataset shape {}'.format(Counter(X_label)))

    ros = RandomOverSampler(random_state=shuffle_stat)
    over_samples_X, over_samples_Y = ros.fit_resample(X_train, X_label)
    print("After OverSampling, counts of label '1': {}".format(sum(over_samples_Y == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(over_samples_Y == 0)))
    return over_samples_X, over_samples_Y


def random_under_sampling(X_train, X_label):
    X_train = array(X_train).astype(float)
    X_label = array(X_label).astype(float)
    print 'Original data set shape {}'.format(Counter(X_label))

    rus = RandomUnderSampler(random_state=shuffle_stat)
    under_samples_X, under_samples_Y = rus.fit_resample(X_train, X_label)
    print "After UnderSampling, counts of label '1': {}".format(sum(under_samples_Y == 1))
    print "After UnderSampling, counts of label '0': {}".format(sum(under_samples_Y == 0))
    return under_samples_X, under_samples_Y


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


def standardize_data(scaler_ica, x_single):
    return scaler_ica.transform(x_single)


if __name__ == '__main__':
    # start
    # print 'training set size', training_set_size

    graph_path = "graph/new_friendship_connected.pickle"
    graph = nx.read_gpickle(graph_path)

    for training_set_size in training_set_list:
        # split into train set and test set
        # X_train, X_test, Y_train, Y_test = split_trainset_testset(graph, attributes_name)
        X_train, X_test, Y_train, Y_test = load_trainset_testset(training_set_size)
        Y_train = array(Y_train).astype(float)
        Y_test = array(Y_test).astype(float)

        print '-----------------------------------------------------------'
        print 'training set', len(X_train)
        print 'test set', len(X_test)

        print "training set '1': {}".format(sum(Y_train == 1))
        print "test set '1' : {}".format(sum(Y_test == 1))
        print 'label 1 ratio: ', float(sum(Y_train == 1)) / (sum(Y_train == 1) + sum(Y_test == 1))

        print "training set '0': {}".format(sum(Y_train == 0))
        print "test set '0' : {}".format(sum(Y_test == 0))
        print 'label 0 ratio: ', float(sum(Y_train == 0)) / (sum(Y_train == 0) + sum(Y_test == 0))
        print '-----------------------------------------------------------'
        # remove label of nodes in test set
        current_graph = remove_test_label(graph, X_test)

        # processing the training set
        zero_zero_num = 0  # record the number of node whose neighbors are all belonged to test set

        for l in X_train:
            node_id = l[0]

            number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)

            # spammers_percentage, non_spammers_persentage = aggregation_percentage(
            # number_of_spammers,
            # number_of_non_spammers
            # )

            # number_of_second_spammers, number_of_second_non_spammer = compute_second_degree_attributes(
            # current_graph,
            # node_id
            # )

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

        # standardize
        if standardize_flag == 1:
            scaler_ica = StandardScaler()
            scaler_ica.fit(X_train_without_id)
            X_train_without_id = standardize_data(scaler_ica, X_train_without_id)
            # print X_train_without_id

        # over/under sampling
        # X_sampling, Y_sampling = random_under_sampling(X_train_without_id, Y_train)

        # train the classifier
        print
        print 'dimension of training features', dimension_of_attr
        # classifier = tree.DecisionTreeClassifier(random_state=shuffle_stat)
        # classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000, C=0.5)
        classifier = GaussianNB()
        # classifier = RandomForestClassifier()

        # if don't use over sampling
        # cross validate
        validate_score = cross_val_score(classifier, X_train_without_id, Y_train,
                                         cv=10)
        print 'cross validate score', validate_score
        classifier.fit(X_train_without_id, Y_train)
        # if use over/under sampling
        # cross validate
        # validate_score = cross_val_score(classifier, X_sampling, Y_sampling,
        #                                  cv=10)
        # print 'cross validate score', validate_score
        # classifier.fit(X_sampling, Y_sampling)
        # print 'features importance', classifier.feature_importances_

        # save the classifier in pickle
        # filename = 'decision_tree_result/decision_tree_model%d.sav' % shuffle_stat
        # pickle.dump(classifier, open(filename, 'wb'))

        # load classifier
        # model_path = 'decision_tree_result/decision_tree_model42.sav'
        # classifier = pickle.load(open(model_path, 'rb'))

        # check the prediction result on training set
        Y_train_predict = list()
        for l in X_train:
            single_data = array(l[1:]).astype('float64').reshape(1, len(l[1:]))
            Y_train_predict.append(classifier.predict(single_data))
            # Y_train_predict.append(classifier.predict([l[1:]]))
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

            # standardize
            if standardize_flag == 1:
                X_test_without_id = standardize_data(scaler_ica, [X_test_without_id])

            X_test_fit = array(X_test_without_id).astype(float)
            Y_ = classifier.predict(X_test_fit.reshape(1, dimension_of_attr))
            Y_predict.append(Y_)

            # update
            current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
            current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
            current_graph.node[node_id]['fake'] = int(Y_)
        report = classification_report(Y_test, Y_predict, digits=6)
        classification_report_csv(report, training_set_size, 'compared ' + classifier_name)
        print classification_report(Y_test, Y_predict, digits=6)
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

            # validate
            # for i, X in enumerate(X_test):
            #     print X
            #     print current_graph.node[X[0]]
            #     if i > 5:
            #         break

            for X_single in X_test:
                X = copy.deepcopy(X_single)
                node_id = X[0]

                # compute attributes
                number_of_spammers, number_of_non_spammers = compute_attribute(current_graph, node_id)
                # spammers_percentage, non_spammers_persentage = aggregation_percentage(number_of_spammers,
                #                                                                       number_of_non_spammers)

                # number_of_second_spammers, number_of_second_non_spammer = compute_second_degree_attributes(
                # current_graph,
                # node_id)
                
                X.append(number_of_spammers)
                X.append(number_of_non_spammers)
                # X.append(number_of_second_spammers)
                # X.append(number_of_second_non_spammer)
                X_test_without_id = X[1:]

                # standardize
                if standardize_flag == 1:
                    X_test_without_id = standardize_data(scaler_ica, [X_test_without_id])

                X_test_fit = array(X_test_without_id).astype(float)
                label_predict = classifier.predict(X_test_fit.reshape(1, dimension_of_attr))
                Y_predict.append(int(label_predict))
                Y_all_predict[X[0]].append(int(label_predict))

                # update
                current_graph.node[node_id]['spammer_neighbors_num'] = number_of_spammers
                current_graph.node[node_id]['non_spammer_neighbors_num'] = number_of_non_spammers
                current_graph.node[node_id]['fake'] = int(label_predict)

            acc = accuracy_score(Y_test, Y_predict)
            report = classification_report(Y_test, Y_predict, digits=6)
            print classification_report(Y_test, Y_predict, digits=6)
        classification_report_csv(report, training_set_size, classifier_name)