# -*- coding: utf-8 -*-
import networkx as nx
import collections
import matplotlib.pyplot as plt
graph_path = "graph/high_degree_partition_2.pickle"
G = nx.read_gpickle(graph_path)

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