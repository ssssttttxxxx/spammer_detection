# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from ica_process import compute_attribute
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

s_times = 5
run_times = 5
iteration = 5
shuffle_stat = 56
training_set_size = 0.5


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


attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num']
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

graph_path = "graph/high_degree_partition_2.pickle"
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
print train_x[:, 1:]

# train the decision tree
classifier_dtree = tree.DecisionTreeClassifier(random_state=shuffle_stat)
classifier_dtree.fit(train_x[:, 1:], train_y)

# predict the label of test set
predict_y = classifier_dtree.predict(test_x[:, 1:])
print classification_report(test_y, predict_y)
combine_test_set = np.column_stack((test_x, predict_y.T))

# assign the label on current graph
for x_pre in combine_test_set:
    if int(x_pre[6]) > 1:
        print 'fucking error'
        print x_pre[6]
    current_graph.node[x_pre[0]]['fake'] = x_pre[6]

# burn-in
test_set = np.column_stack((test_x, test_y))
print test_set.shape

original_len = test_set.shape[1]
print 'original', original_len

for iteration in range(s_times):

    np.random.shuffle(test_set)
    x_t = test_set[:, 1:6]
    y_t = np.copy(test_set[:, 6]).astype(np.int8)
    y_predict = classifier_dtree.predict(x_t)
    # todo: predict一次更新一次
    print y_t.shape
    print classification_report(y_t, y_predict)

    # combine into the last column in the test set (np.array)
    test_set = np.column_stack((test_set, y_predict))

    # update the label in graph
    for x_label_pre in test_set:
        current_graph.node[x_label_pre[0]]['fake'] = x_label_pre[original_len+iteration]

# initialize sample counts with numpy
test_set = np.column_stack((test_x, test_y))
print 'init', test_set.shape

# collect samples
original_len = test_set.shape[1]
print 'original', original_len

for iteration in range(run_times):

    np.random.shuffle(test_set)
    x_t = test_set[:, 1:6]
    y_t = np.copy(test_set[:, 6]).astype(np.int8)
    y_predict = classifier_dtree.predict(x_t)
    # todo: predict一次更新一次
    print y_t.shape
    print classification_report(y_t, y_predict)

    # combine into the last column in the test set (np.array)
    test_set = np.column_stack((test_set, y_predict))

    # update the label in graph
    for x_label_pre in test_set:
        current_graph.node[x_label_pre[0]]['fake'] = x_label_pre[original_len+iteration]


node_ids = np.copy(test_set[:, 0])
labels_distribution = np.copy(test_set[:, 7:])
