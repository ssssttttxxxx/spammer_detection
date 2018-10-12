# -*- coding: utf-8 -*-
import networkx as nx

"""
After building the graph,
After labeling the nodes,
After removing unknown nodes,
After removing nodes with degree = 0,
"""


class IterativeClassification:
    def __init__(self, graph_path):
        self.graph = nx.read_gpickle(graph_path)
        pass

    def devide_train_test(self):
        total_list = list()
        for i, node in enumerate(self.graph.nodes()):
            temp_list = list()
            for attr in self.graph.node[node].items():
                temp_list.append(attr)
            total_list.append(temp_list)
        pass

    def train_classifier(self):
        pass

    def generate_random_order(self):
        pass

    def compute_attributes(self):
        pass

    def calculate_accuracy(self):
        pass

    def store_result(self):
        pass



# graph with label
graph_path = 'friendship_graph/friendship_reviewer_label771487.pickle'
graph = nx.read_gpickle(graph_path)

# calculate the relational attributes
for iter, node in enumerate(graph.nodes()):

    neighbors = graph.neighbors(node)
    number_of_spammers = 0
    number_of_non_spammers = 0

    for neighbor in neighbors:
        if graph.node[neighbor]['fake'] == 1:
            number_of_spammers += 1
        else:
            number_of_non_spammers += 1

    graph.node[node]['spammer_neighbors_num'] = number_of_spammers
    graph.node[node]['non_spammer_neighbors_num'] = number_of_non_spammers
    print graph.node[node]

nx.write_gpickle(graph, 'friendship_graph/friendship_network_attr_reviewer_lable771487.pickle')
