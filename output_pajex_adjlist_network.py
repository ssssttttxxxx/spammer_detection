# -*- coding: utf-8 -*-

import networkx as nx

graph_path = 'graph/old_friendship_connected.pickle'
graph = nx.read_gpickle(graph_path)

# nx.write_pajek(graph, 'pajek/separate_spammer_pajek.net')
# nx.write_pajek(graph, 'pajek/separate_spammer_771487.net')
nx.write_edgelist(graph, 'graph/old_friendship_connected.edgelist')