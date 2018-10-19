# -*- coding: utf-8 -*-
import networkx as nx
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

trainset_size = 0.5
run_times = 2
iterations = 20
shuffle_stat = 42
attributes_name = ['reviewerID', 'friends_num', 'reviews_num', 'photo_num']
# attributes_name = ['reviewerID',]


#  divide data by training set and test set
def split_tarinset_testset(graph, attributes):
    X_list = list()
    Y_list = list()
    for i, node in enumerate(graph.nodes()):
        temp_list = list()
        for attr_name, val in graph.node[node].items():

            if attr_name in attributes:
                temp_list.append(val)
            elif attr_name == 'fake':
                Y_list.append(val)

        X_list.append(temp_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size, random_state=42)
    return X_train, X_test, Y_train, Y_test


#  remove the test set label on the graph data
def remove_test_label(graph, delete_list):
    # print "remove test set label"
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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

X_all_train = list()
X_all_test = list()

Y_all_test = list()
X_all_train_real = list()

Y_all_predict = list()


for round_num in range(run_times):

    X_train, X_test, Y_train, Y_test = split_tarinset_testset(graph, attributes_name)

    # remove label of nodes in test set
    current_graph = remove_test_label(graph, X_test)

    X_train_without_id = [node[1:] for node in X_train]

    classifier = tree.DecisionTreeClassifier(criterion="entropy")
    classifier.fit(X_train_without_id, Y_train)

    Y_predict = list()
    for X in X_test:
        # X = copy.deepcopy(X_single)
        label_predict = classifier.predict([X[1:]])
        Y_predict.append(int(label_predict))

    Y_all_predict.append(Y_predict)
    Y_all_test.append(Y_test)
    X_all_train.append(X_train_without_id)
    X_all_train_real.append(X_train)
    X_all_test.append(X_test)

    # print current_graph.node[node_id]
    micro = f1_score(Y_test, Y_predict, average='micro')
    macro = f1_score(Y_test, Y_predict, average='macro')
    recall_rate = recall_score(Y_test, Y_predict, average='binary')
    print "f1 macro is", macro
    print "f1 micro is", micro
    print "recall rate is", recall_rate

if len(Y_all_predict) > 1:
    if Y_all_predict[1] == Y_all_predict[0]:
        print 'Y predict same'
    else:
        print 'Y predict not same'

    if Y_all_test[1] == Y_all_test[0]:
        print 'Y test same'
    else:
        print 'Y test not same'

    if X_all_test[1] == X_all_test[1]:
        print 'X train same'
    else:
        print 'X train not same'

    if X_all_train_real[1] == X_all_train_real[0]:
        print 'X train without id same'
    else:
        print 'X train without id not same'

    if X_all_train[1] == X_all_train[0]:
        print 'X train same'
    else:
        print 'X train not same'
