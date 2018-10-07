# -*- coding: utf-8 -*-

import networkx as nx

graph_path = 'graph/separate_spammer_771000.pickle'
graph = nx.read_gpickle(graph_path)

# nx.write_pajek(graph, 'pajek/separate_spammer_771000_pajek.net')
nx.write_gexf(graph, 'gexf/separate_spammer_771000.gexf')