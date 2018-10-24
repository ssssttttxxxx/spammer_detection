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

trainset_size = 0.5
shuffle_stat = 42

def split_trainset_testset(graph):

    X_list = list()
    Y_list = list()

    with open('embeddings/friendship_reviewer_label_attr_clean_unknown_degree0.embeddings', 'r') as embeddings:
        summary = embeddings.readline().split()
        num_of_nodes = summary[0].strip()
        num_of_dimension = summary[1].strip()

        print summary
        for line in embeddings:
            print line

            break

    # print "split train set and test set "
    # print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    #
    # X_list = list()
    # Y_list = list()
    # for i, node in enumerate(graph.nodes()):
    #     temp_list = list()
    #     for attr_name, val in graph.node[node].items():
    #
    #         if attr_name in attributes:
    #             temp_list.append(val)
    #         elif attr_name == 'fake':
    #             Y_list.append(val)
    #         elif attr_name == 'degree':
    #             degree = graph.degree(node)
    #             Y_list.append(int(degree))
    #
    #     X_list.append(temp_list)

    # X_train, X_test, Y_train, Y_test = train_test_split(X_list, Y_list, test_size=1-trainset_size, random_state=shuffle_stat)

if __name__ == '__main__':
    split_trainset_testset()