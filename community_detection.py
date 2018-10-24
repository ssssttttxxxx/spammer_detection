# -*- coding: utf-8 -*-
import time
import community
import networkx as nx
import matplotlib.pyplot as plt

print nx.__version__
print community.__version__

graph_path = 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle'
graph = nx.read_gpickle(graph_path)

# part = community.best_partition(graph)
# 计算模块度
# mod = community.modularity(part,graph)
# 绘图

# values = [part.get(node) for node in graph.nodes()]
# nx.draw_spring(graph, cmap=plt.get_cmap('jet'), node_color=values, node_size=30, with_labels=False)
# plt.show()

k = 4
print 'community detecting'
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# partitions_generator = nx.algorithms.community.k_clique_communities(graph, k)
partitions_generator = nx.algorithms.community.label_propagation_communities(graph)


print 'finished'
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
partitions = list(partitions_generator)

print 'convert to list '
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
print len(partitions)

for nodes in partitions:
    color_list = list()
    # print nodes
    for node in nodes:
        if graph.node[node]['fake'] == 1:
            color_list.append('r')
        else:
            color_list.append('b')
    h = graph.subgraph(nodes)
    pos = nx.spring_layout(h)
    nx.draw_networkx_nodes(h, pos,
                  node_list=nodes,
                  node_color=color_list,
                  node_size=200,)
    edges = h.edges()
    nx.draw_networkx_edges(h, pos, edges_list=edges, width=1)
    # nx.draw(h, pos, cmap=plt.get_cmap('jet'))
    plt.show()