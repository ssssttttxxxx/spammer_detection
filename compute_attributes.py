# -*- coding: utf-8 -*-
import networkx as nx


def delete_lable(graph, delete_list):
    current_graph = graph.copy()
    for node in delete_list:
        current_graph.node[node]['fake'] = 'unknown'
        # current_graph.node[node]['fake'] = 0
        return current_graph


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

        # print current_graph.node[node]

