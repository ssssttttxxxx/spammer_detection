# -*- coding: utf-8 -*-

import networkx as nx

graph_path = '../graph/reduce_clean.pickle'
graph = nx.read_gpickle(graph_path)

# nx.write_pajek(graph, 'pajek/separate_spammer_771000_pajek.net')
nx.write_adjlist(graph, '../graph/reduce_clean.adjlist')
