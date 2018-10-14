# -*- coding: utf-8 -*-
from sklearn import tree
import sklearn
import networkx as nx
import graphviz


print nx.__version__
graph_path = 'graph/friendship_attr_total_cleanUnknown.pickle'
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

def compute_attribute(current_graph, classifier):
    for node in current_graph.nodes():
        neighbors = current_graph.neighbors(node)
        number_of_spammers = 0
        number_of_non_spammers = 0

        for neighbor in neighbors:
            if current_graph.node[neighbor]['fake'] == 1:
                number_of_spammers += 1
            elif current_graph.node[neighbor]['fake'] == 0:
                number_of_non_spammers += 1

        current_graph.node[node]['spammer_neighbors_num'] = number_of_spammers
        current_graph.node[node]['non_spammer_neighbors_num'] = number_of_non_spammers

        """
        Then classifier predict the label of the current node
        """
        node_attr = list()
        for val in current_graph.node[node].values():
            node_attr.append(val)

        lable = classifier.predict(node_attr)[0][0]