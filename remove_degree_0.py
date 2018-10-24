# -*- coding: utf-8 -*-
import networkx as nx
import os
import time
"""
for network/graph with nodes do not in the database, remove them from the network/graph.
"""

graph_path = 'graph/reduce_label_attr_clean_unknown_degree0.pickle'
graph = nx.read_gpickle(graph_path)

remove_list = list()

for iter, node in enumerate(list(graph.nodes())):
    # print graph.node[node]

    flag = graph.node[node]['friends_num']

    if graph.degree(node) == 0:
        # print graph.degree(node)
        remove_list.append(node)
    if iter % 1000 == 0:
        print iter, len(remove_list)
        print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        # nx.write_gpickle(graph, 'graph/friendship_attr_cleanUnknown_%s.pickle' % iter)
        # if iter != 0:
        #     os.remove( 'graph/friendship_attr_cleanUnknown_%s.pickle' % (iter - 1000))

for node in remove_list:
    graph.remove_node(node)
print iter
nx.write_gpickle(graph, 'graph/reduce_clean.pickle')