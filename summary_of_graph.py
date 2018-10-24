# -*- coding: utf-8 -*-
import networkx as nx

if __name__ == "__main__":
    graph_path = 'graph/separate_spammer_771487.pickle'
    graph_attr_path = 'graph/reduce_label_attr_clean_unknown_degree0.pickle'
    friendship_graph = nx.read_gpickle(graph_path)
    friendship_graph_attr = nx.read_gpickle(graph_attr_path)
    print "clean unknown"
    print friendship_graph.number_of_nodes()
    print friendship_graph.number_of_edges()
    print "reduce clean unknown and degree_0"
    print friendship_graph_attr.number_of_nodes()
    print friendship_graph_attr.number_of_edges()

    num_spammer = 0
    num_normal = 0
    for n in friendship_graph_attr:
        if friendship_graph_attr.node[n]['fake'] == 1:
            num_spammer += 1
        elif friendship_graph_attr.node[n]['fake'] == 0:
            num_normal += 1

    print "spammers", num_spammer
    print 'normal', num_normal