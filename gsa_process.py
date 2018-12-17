# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from collections import Counter
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from read_training_test_set import read_data_from_file

s_times = 5
run_times = 20
iteration = 5
shuffle_stat = 42
training_set_size = 0.5
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num']


def compute_attribute(current_graph, node):
    """
    compute attributes according to current graph
    :param current_graph:
    :param node:
    :return:
    """
    neighbors = current_graph.neighbors(node)
    number_of_spammers = 0
    number_of_non_spammers = 0

    for neighbor in neighbors:
        if int(current_graph.node[neighbor]['fake']) == 1:
            number_of_spammers += 1
        elif int(current_graph.node[neighbor]['fake']) == 0:
            number_of_non_spammers += 1
        else:
            continue
    return number_of_spammers, number_of_non_spammers


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
    current_graph = graph.copy()
    for node in delete_list:
        current_graph.node[node[0]]['fake'] = 2
        # print 'remove label of %s' % node[0]
    return current_graph


def most_common(lst):
    """
    return the most common element in the list
    :param lst:
    :return:
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


# graph_path = "graph/high_degree_partition_2.pickle"
graph_path = "graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle"
graph = nx.read_gpickle(graph_path)

x_train, x_test, y_train, y_test = split_trainset_testset(graph, attributes_name)
current_graph = remove_test_label(graph, x_test)

# process the data to in numpy
train_x = np.array(x_train)
train_y = np.array(y_train)
test_x = np.array(x_test)
test_y = np.array(y_test)

# bootstrapping
relation_attr_list = list()
for node_id in train_x[:, 0]:
    spammer_num, non_spammer_num = compute_attribute(current_graph, node_id)
    current_graph.node[node_id]['num_of_spammers'] = spammer_num
    current_graph.node[node_id]['num_of_non_spammers'] = non_spammer_num
    relation_attr_list.append([spammer_num, non_spammer_num])

relation_attr = np.array(relation_attr_list)
train_x = np.concatenate((train_x, relation_attr), axis=1)

relation_attr_list = list()
for node_id in test_x[:, 0]:
    spammer_num, non_spammer_num = compute_attribute(current_graph, node_id)
    current_graph.node[node_id]['num_of_spammers'] = spammer_num
    current_graph.node[node_id]['num_of_non_spammers'] = non_spammer_num
    relation_attr_list.append([spammer_num, non_spammer_num])

relation_attr = np.array(relation_attr_list)
test_x = np.concatenate((test_x, relation_attr), axis=1)

# shape of training data
print train_x.shape
print train_y.shape
print test_x.shape
print test_y.shape

# train the decision tree
classifier_dtree = tree.DecisionTreeClassifier(random_state=shuffle_stat)
classifier_dtree.fit(train_x[:, 1:], train_y)
print '################### classifier result #####################'
train_y_ = classifier_dtree.predict(train_x[:, 1:])
print classification_report(train_y, train_y_)


# predict the label of test set
predict_y = classifier_dtree.predict(test_x[:, 1:])
print classification_report(test_y, predict_y)
combine_test_set = np.column_stack((test_x, predict_y.T))

# assign the label on current graph
for x_pre in combine_test_set:
    if int(x_pre[6]) > 1:
        print 'error'
        print x_pre[6]

    # print 'before assignment', current_graph.node[x_pre[0]]
    current_graph.node[x_pre[0]]['fake'] = int(x_pre[6])
    # print 'after assignment', current_graph.node[x_pre[0]]


# burn-in
print '###################### burn-in ###########################'
test_set = np.column_stack((test_x, test_y))
print test_set.shape

original_len = test_set.shape[1]
print 'original', original_len

for iteration in range(s_times):

    # generate random order
    np.random.shuffle(test_set)

    # update attributes of node
    for t in test_set:
        node_id = t[0]
        spammer_num, legitimate_num = compute_attribute(current_graph, node_id)
        t[4] = spammer_num
        t[5] = legitimate_num

    x_t = test_set[:, 1:6]
    y_t = np.copy(test_set[:, 6]).astype(np.int8)
    y_predict = classifier_dtree.predict(x_t)
    # todo: predict一次更新一次
    print y_t.shape
    print classification_report(y_t, y_predict, digits=6)

    # combine into the last column in the test set (np.array)
    test_set = np.column_stack((test_set, y_predict))

    # update the label in graph
    for x_label_pre in test_set:
        current_graph.node[x_label_pre[0]]['fake'] = x_label_pre[original_len+iteration]

# initialize sample counts with numpy
test_set = np.column_stack((test_x, test_y))
print 'init', test_set.shape

# collect samples
print '######################## collect samples #########################'

original_len = test_set.shape[1]

for iteration in range(run_times):

    np.random.shuffle(test_set)

    # update attributes of node
    for t in test_set:
        node_id = t[0]
        spammer_num, legitimate_num = compute_attribute(current_graph, node_id)
        t[4] = spammer_num
        t[5] = legitimate_num

    x_t = test_set[:, 1:6]
    y_t = np.copy(test_set[:, 6]).astype(np.int8)
    y_predict = classifier_dtree.predict(x_t)
    # todo: predict一次更新一次
    print y_t.shape
    print classification_report(y_t, y_predict, digits=6)

    # combine into the last column in the test set (np.array)
    test_set = np.column_stack((test_set, y_predict))

    # update the label in graph
    for x_label_pre in test_set:
        current_graph.node[x_label_pre[0]]['fake'] = x_label_pre[original_len+iteration]

node_ids = np.copy(test_set[:, 0])
real_label = np.copy(test_set[:, 6].astype(np.int8))
labels_distribution = np.copy(test_set[:, 7:].astype(np.int8))
node_label_dist = np.column_stack((node_ids, labels_distribution))

node_label_predict = list()
for r in node_label_dist:
    node_label_predict.append(int(most_common(r[1:])))

print '########################### final result #############################'
print classification_report(real_label, node_label_predict, digits=6)
