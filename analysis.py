# -*- coding: utf-8 -*-
import networkx as nx
import collections
import matplotlib.pyplot as plt


def draw_bar(degree_count):
    """
    Parameters
    ----------
    degree_count: list or sequence, uncertain

    Returns
    -------

    """
    deg, cnt = zip(*degree_count.items())

    count_ = list()
    degree_ = list()
    count_100 = 0
    for d, c in zip(deg, cnt):
        if d > 100:
            count_100 += c
        else:
            count_.append(c)
            degree_.append(d)
    degree_.append(100)
    count_.append(count_100)

    print len(count_)
    print len(degree_)

    plt.bar(degree_, count_, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.show()


def plt_bar_for_all_degree(G):
    """

    Returns
    -------

    """
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    count_ = list()
    degree_ = list()
    count_100 = 0
    for d, c in zip(deg, cnt):
        if d > 100:
            count_100 += c
        else:
            count_.append(c)
            degree_.append(d)
    degree_.append(100)
    count_.append(count_100)

    print len(count_)
    print len(degree_)

    plt.bar(degree_, count_, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")

    # draw graph in inset
    # Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    # pos = nx.spring_layout(G)
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()


def plt_bar_division(G):
    """

    Parameters
    ----------
    G: graph

    Returns
    -------

    """
    # divede the spammer and legitimate user
    spammer_list = list()
    legitimate_list = list()
    for node in G.nodes():
        fake_flag = G.node[node]['fake']
        if fake_flag == 1:
            degree = G.degree(node)
            spammer_list.append(degree)
        elif fake_flag == 0:
            degree = G.degree(node)
            legitimate_list.append(degree)

    # count the frequency
    spammer_list.sort()
    spammer_degree_count = collections.Counter(spammer_list)
    print spammer_degree_count
    legitimate_list.sort()
    legitimate_degree_count = collections.Counter(legitimate_list)
    print legitimate_degree_count

    # draw bar graph
    draw_bar(spammer_degree_count)
    draw_bar(legitimate_degree_count)

if __name__ == '__main__':
    graph_path = "graph/high_degree_partition_2.pickle"
    G = nx.read_gpickle(graph_path)
    plt_bar_division(G)
