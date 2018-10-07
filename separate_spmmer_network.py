# -*- coding: utf-8 -*-
import networkx as nx
import time
import os
class SeparateNetwork():

    def __init__(self):
        print "begin init", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.graph_path = 'friendship_graph/friendship_reviewer_label771487.pickle'
        self.friendship_graph = nx.read_gpickle(self.graph_path)
        print self.friendship_graph.number_of_edges()
        print "end of init", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    def separate(self):
        record_point = 0
        if record_point == 0:
            spammmer_graph = nx.Graph()
            non_spammer_graph = nx.Graph()
        else:
            spammmer_graph = nx.Graph("graph/separate_spammer_%d.pickle" % record_point)
            non_spammer_graph = nx.Graph("graph/separate_non_spammer_%d.pickle" % record_point)

        for num, node in enumerate(self.friendship_graph.nodes()):
            fake = self.friendship_graph.node[node]['fake']
            if fake == 1:
                spammmer_graph.add_node(node, reviewerID=node, fake=1)
                neighbors = self.friendship_graph.neighbors(node)
                # iterate the neighbors of node and add in the graph
                for neighbor in neighbors:
                    neighbor_fake = self.friendship_graph.node[node]['fake']
                    if neighbor_fake == 1:
                        spammmer_graph.add_node(neighbor, reviewerID=neighbor, fake=1)
                        spammmer_graph.add_edge(node, neighbor)
            else:
                non_spammer_graph.add_node(node, reviewerID=node, fake=0)
                neighbors = self.friendship_graph.neighbors(node)
                # iterate the neighbors of node and add in the graph
                for neighbor in neighbors:
                    neighbor_fake = self.friendship_graph.node[node]['fake']
                    if neighbor_fake == 0:
                        non_spammer_graph.add_node(neighbor, reviewerID=neighbor, fake=0)
                        non_spammer_graph.add_edge(node, neighbor)

            if num % 1000 == 0:
                nx.write_gpickle(spammmer_graph, "graph/separate_spammer_%d.pickle" % num)
                nx.write_gpickle(non_spammer_graph, "graph/separate_non_spammer_%d.pickle" % num)
                print num, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                # delete the old pickle
                if num != 0:
                    os.remove("graph/separate_spammer_%d.pickle" % (num-1000))
                    os.remove("graph/separate_non_spammer_%d.pickle" % (num-1000))
        # for the rest node less than 1000
        nx.write_gpickle(spammmer_graph, "graph/separate_spammer_%d.pickle" % num)
        nx.write_gpickle(non_spammer_graph, "graph/separate_non_spammer_%d.pickle" % num)

if __name__ == "__main__":
    print nx.__version__
    SN = SeparateNetwork()
    SN.separate()