# -*- coding: utf-8 -*-
import MySQLdb
import networkx as nx
# import matplotlib.pyplot as plt
import os
import time

class ReadFriendshipGraph():

    def __init__(self):
        print "begin init", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.graph_path = 'friendship_graph/friendship_925100.pickle'
        self.friendship_graph = nx.read_gpickle(self.graph_path)
        self.db = MySQLdb.connect("quantum2.is.cityu.edu.hk.", "readyelp", "yelp2018", "yelp", charset="utf8")
        self.cursor = self.db.cursor()
        print "end of init", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    def read_friendship_graph(self):
        """
        read the graph data in pickle format
        :return: nx.graph
        """
        graph = nx.read_gpickle(self.graph_path)
        return graph

    def determine_spammer(self, reviewer_id):
        """
        whether the reviewer is a spammer
        :param reviewer_id:
        :return:
        """
        fake_sql = "select * from reviews where reviewerID = '%s' and fake = 1" % reviewer_id
        self.cursor.execute(fake_sql)
        data = self.cursor.fetchall()

        # 0 represents normal user
        if len(data) == 0:
            return 0
        # 1 represents spammer
        else:
            return 1

    def label_users(self):
        print self.friendship_graph.number_of_nodes()
        for node in self.friendship_graph.nodes():
            fake_flag = self.determine_spammer(node)
            self.friendship_graph.node[node]['fake'] = fake_flag
            print self.friendship_graph.node[node]

if __name__ == "__main__":
    RFG = ReadFriendshipGraph()
    # flag = RFG.determine_spammer('DFWXDnzAyxQ7kxg12h4BlQ')
    # print flag

    RFG.label_users()