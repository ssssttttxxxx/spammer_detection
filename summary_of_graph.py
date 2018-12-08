# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    graph_path_1 = 'graph/old_friendship_connected.pickle'
    # graph_path_2 = 'graph/high_degree_partition_from_friendship_attr_total_cleanUnknown_cleanDegree0.pickle'
    graph_1 = nx.read_gpickle(graph_path_1)
    # graph_2 = nx.read_gpickle(graph_path_2)
    print "graph 1"
    print 'nodes', graph_1.number_of_nodes()
    print 'edges', graph_1.number_of_edges()
    # print "graph 2"
    # print graph_2.number_of_nodes()
    # print graph_2.number_of_edges()

    num_spammer = 0
    num_normal = 0
    for n in graph_1:
        if graph_1.node[n]['fake'] == 1:
            num_spammer += 1
            if num_spammer < 5:
                print 'spammer example', num_spammer, graph_1.node[n]

        elif graph_1.node[n]['fake'] == 0:
            num_normal += 1

    print '1 is connected ?', nx.is_connected(graph_1)

    print "spammers", num_spammer
    print 'non-spammers', num_normal
    print list(graph_1.nodes(data=True))[0]
    # print 'attributes', graph_1.node[graph_1.nodes()[0]]
    degree_sequence = sorted([d for n, d in graph_1.degree()], reverse=True)
    degree_max = max(degree_sequence)
    print 'degree max', degree_max

    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    #
    # # draw graph in inset
    # plt.axes([0.45, 0.45, 0.45, 0.45])
    # Gcc = sorted(nx.connected_component_subgraphs(graph_2), key=len, reverse=True)[0]
    # pos = nx.spring_layout(Gcc)
    # plt.axis('off')
    # nx.draw_networkx_nodes(Gcc, pos, node_size=20)
    # nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
    #
    # plt.show()

