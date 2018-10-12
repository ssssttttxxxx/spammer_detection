# -*- coding: utf-8 -*-
from sklearn import tree
import sklearn
import networkx as nx
import graphviz

graph_path = 'graph/friendship_attr_total_cleanUnknown_cleanDegree0.pickle'
graph_label_path = 'friendship_graph/friendship_925100.pickle'
graph = nx.read_gpickle(graph_path)


class DecisionTreeForGraph:

    def __init__(self):
        pass

    def train_decision_tree(self, X_array, Y_array):
        classifier = tree.DecisionTreeClassifier(criterion="entropy")
        classifier.fit(X_array, Y_array)
        tree_data = tree.export_graphviz(classifier, out_file=None)
        tree_graph = graphviz.Source(tree_data)
        tree_graph.render("decision_tree_result/spammer_decision")


if __name__ == "__main__":

    X_array = list()
    for i, node in enumerate(graph.nodes()):
        if i == 20:
            break
        temp_list = list()
        print graph.node[node]
        temp_list.append(i)
        for attribute_num, val in enumerate(graph.node[node].values()):
            if attribute_num == 0 or attribute_num == 1 or attribute_num == 3:
                continue
            else:
                temp_list.append(val)
        X_array.append(temp_list)

    print len(X_array)
    print X_array
    Y_array = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    print len(Y_array)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_array, Y_array)

    predict_item = [[0, 31L, 1L, 2L]]
    print clf.predict(predict_item)

    # with open("decision_tree_example_model.dot", 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f)

    dot_data = tree.export_graphviz(clf, out_file=None)
    decision_tree_graph = graphviz.Source(dot_data)
    decision_tree_graph.render("decision_tree_result/example")
