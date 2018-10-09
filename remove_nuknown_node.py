# -*- coding: utf-8 -*-
import networkx as nx
import os
import time
"""
for network/graph with nodes do not in the database, remove them from the network/graph.
"""

graph_path = 'friendship_graph/friendship_attr_925100.pickle'
graph = nx.read_gpickle(graph_path)

for iter, node in enumerate(graph.nodes()):

    flag = graph.node[node]['friends_num']
    if flag == 'unknow':
        graph.remove_node(node)

    if iter % 1000 == 0:
        print iter
        print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        nx.write_gpickle(graph, 'graph/friendship_attr_cleanUnknown_%s.pickle' % iter)
        if iter != 0:
            os.remove( 'graph/friendship_attr_cleanUnknown_%s.pickle' % (iter - 1000))

nx.write_gpickle(graph, 'graph/friendship_attr_total_cleanUnknown.pickle')