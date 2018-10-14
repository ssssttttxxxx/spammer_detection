# -*- coding: utf-8 -*-
import networkx as nx

graph_path = 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle'
graph = nx.read_gpickle(graph_path)
new_graph = nx.Graph()
node_num = 20000
for node in graph:
    node_detail = graph.node[node]
    # print node_detail
    new_graph.add_node(node, reviewerID=node,
                       reviewerName=node_detail['reviewerName'],
                       location=node_detail['location'],
                       friends_num=node_detail['friends_num'],
                       fake=node_detail['fake'],
                       reviews_num=node_detail['reviews_num'],
                       photo_num=node_detail['photo_num'],
                       )

    neighbors = graph.neighbors(node)
    for neighbor in neighbors:
        print 'neighbor'
        neighbor_detail = graph.node[neighbor]
        new_graph.add_node(neighbor, reviewerID=neighbor,
                           reviewerName=neighbor_detail['reviewerName'],
                           location=neighbor_detail['location'],
                           friends_num=neighbor_detail['friends_num'],
                           fake=neighbor_detail['fake'],
                           reviews_num=neighbor_detail['reviews_num'],
                           photo_num=neighbor_detail['photo_num'],
                           )
        new_graph.add_edge(node, neighbor)
    if new_graph.number_of_nodes() > node_num:
        break

for node in new_graph:
    non_spammer = 0
    spammer = 0
    if new_graph.node[node]['fake'] == 1:
        spammer += 1
    else:
        non_spammer += 1
    print 'spammer', spammer, 'non-spammer', non_spammer

nx.write_gpickle(new_graph, 'graph/reduce_label_attr_clean_unknown_degree0.pickle')