# -*- coding: utf-8 -*-
import networkx as nx

if __name__ == "__main__":
    graph_path = 'friendship_graph/friendship_925100.pickle'
    graph_attr_path = 'friendship_graph/friendship_attr_925100.pickle'
    friendship_graph = nx.read_gpickle(graph_path)
    friendship_graph_attr = nx.read_gpickle(graph_attr_path)
    print "friendship"
    print friendship_graph.number_of_nodes()
    print friendship_graph.number_of_edges()
    print "friendship_attr"
    print friendship_graph_attr.number_of_nodes()
    print friendship_graph_attr.number_of_edges()