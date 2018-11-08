# -*- coding: utf-8 -*-
import networkx as nx
import sys
sys.setrecursionlimit(10000)

depth = 0


def construct_high_degree_network_list(graph, node, blank_list, depth):

    depth += 1
    if depth > 100:
        return

    blank_list.append(node)
    neighbors = graph.neighbors(node)
    num_of_neighbors = len(list(neighbors))
    neighbors = graph.neighbors(node)
    count = 0

    for neighbor in neighbors:
        if neighbor not in blank_list:
            blank_list.append(neighbor)

            construct_high_degree_network_list(graph, neighbor, blank_list, depth)
        else:
            count += 1

    if count == num_of_neighbors:
        # print node, 'return'
        return


def create_graph_by_node(graph, node):
    return nx.node_connected_component(graph, node)


graph_path = 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle'
graph = nx.read_gpickle(graph_path)

node_degree_list = [(n, d) for n, d in graph.degree()]
max_degree_node = max(node_degree_list, key=lambda item: item[1])[0]
print 'max', max_degree_node


# construct high degree graph
node_list = list()
construct_high_degree_network_list(graph, max_degree_node, node_list, depth)
# node_list = create_graph_by_node(graph, max_degree_node)
print 'ending'
print len(node_list)
high_connected_graph = graph.subgraph(node_list)
nx.write_gpickle(high_connected_graph, 'graph/high_degree_partition_2.pickle')



