# -*- coding: utf-8 -*-

import networkx as nx

graph_path = 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle'
graph = nx.read_gpickle(graph_path)

# nx.write_pajek(graph, 'pajek/separate_spammer_pajek.net')
# nx.write_pajek(graph, 'pajek/separate_spammer_771487.net')
nx.write_adjlist(graph, 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.adjlist')