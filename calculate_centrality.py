# -*- coding: utf-8 -*-
import networkx as nx
# import MySQLdb
import time


class CalculateCentrality:
    def __init__(self):
        # init graph
        self.spammer_graph_path = 'graph/separate_spammer_771487.pickle'
        self.spammer_graph = nx.read_gpickle(self.spammer_graph_path)
        self.non_spammer_graph_path = 'graph/separate_non_spammer_771487.pickle'
        self.non_spammer_graph = nx.read_gpickle(self.non_spammer_graph_path)

        # init mysql
        # self.db = MySQLdb.connect("localhost", "root", "stx11stx11", "yelp_data", charset="utf8")
        # self.cursor = self.db.cursor()

    # def __del__(self):
    #     self.cursor.commit()

    def summary(self):
        print self.spammer_graph.number_of_nodes()
        print self.spammer_graph.number_of_edges()
        print self.non_spammer_graph.number_of_nodes()
        print self.non_spammer_graph.number_of_edges()

    def degree_centrality(self):

        insert_item_list = list()

        # spammer centrality
        for num, node in enumerate(self.spammer_graph.nodes()):
            item = (node, str(self.spammer_graph.degree(node)))
            insert_item_list.append(item)
            if num % 1000 == 0:
                print num
                self.insert_degree_centrality(insert_item_list)
                insert_item_list = list()
        print num
        self.insert_degree_centrality(insert_item_list)


        # all_node_degree_centrality = nx.degree_centrality(self.spammer_graph)

    def betweeness_centrality(self):

        # for num, node in enumerate(self.spammer_graph.nodes()):
        #     if num == 2:
        #         break
        #     print self.spammer_graph.betweeness(node)
        print "begin betweenness cal", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        all_node_betweenness = nx.betweenness_centrality(self.spammer_graph)
        print "end betweenness cal", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        for iter, item in enumerate(all_node_betweenness):
            if iter == 2:
                break
            print iter
            print item


    def insert_degree_centrality(self, params):
        """
        :param ids: list of id
        :param degree_centralities: list of degree centralities
        :return:
        """
        spammer_sql = "insert into spammer_degree_centrality(id, degree_centrality) values (%s, %s)"
        self.cursor.executemany(spammer_sql, params)
        self.db.commit()

if __name__ == '__main__':
    CC = CalculateCentrality()
    CC.summary()

    # CC.degree_centrality()
    CC.betweeness_centrality()