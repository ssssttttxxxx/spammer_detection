# -*- coding: utf-8 -*-
from devide_train_test import split_tarinset_testset
from sklearn import tree
import sklearn
import networkx as nx
import graphviz


def training_process(graph_path):

    X_train, Y_train, X_test, Y_test = split_tarinset_testset(graph_path)

    classifier = tree.DecisionTreeClassifier(criterion="entropy")
    classifier.fit(X_train, Y_train)
    # visualize decision tree
    tree_data = tree.export_graphviz(classifier, out_file=None)
    tree_graph = graphviz.Source(tree_data)
    tree_graph.render("decision_tree_result/spammer_decision")

    return classifier
