# -*- coding: utf-8 -*-
import networkx as nx

class CalculateCentrality():

    def __init__(self):
        self.spammer_graph_path = 'graph/separate_spammer_771487.pickle'
        self.spammer_graph = nx.read_gpickle(self.spammer_graph_path)
        self.non_spammer_graph_path = 'graph/separate_non_spammer_771487.pickle'
        self.non_spammer_graph = nx.read_gpickle(self.non_spammer_graph_path)

    def summary(self):
        print self.spammer_graph.number_of_nodes()
        print self.spammer_graph.number_of_edges()
        print self.non_spammer_graph.number_of_nodes()
        print self.non_spammer_graph.number_of_edges()

    def centrality(self):

        # spammer centrality
        for num, node in enumerate(self.spammer_graph.nodes()):
            if num == 2:
                break

            print node, self.spammer_graph.node[node]
            for neighbor in self.spammer_graph.neighbors(node):
                print neighbor
            print self.spammer_graph.degree(node)

        # all_node_degree_centrality = nx.degree_centrality(self.spammer_graph)


if __name__ == '__main__':

    CC = CalculateCentrality()
    CC.summary()

    CC.centrality()
