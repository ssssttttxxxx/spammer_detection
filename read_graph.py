# -*- coding: utf-8 -*-
import MySQLdb
import networkx as nx
# import matplotlib.pyplot as plt
import os
import time


class ReadFriendshipGraph:
    """
    This class include the label funciton (为网络中的节点添加标签)
    """

    def __init__(self):
        print "begin init", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.graph_path = 'friendship_graph/friendship_attr_925100.pickle'
        self.friendship_graph = nx.read_gpickle(self.graph_path)
        # self.db = MySQLdb.connect("quantum2.is.cityu.edu.hk.", "readyelp", "yelp2018", "yelp", charset="utf8")
        self.db = MySQLdb.connect("localhost", "root", "stx11stx11", "yelp_data", charset="utf8")

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
        fake_sql = "select * from reviews_simple where reviewerID = '%s' and fake = 1" % reviewer_id
        self.cursor.execute(fake_sql)
        data = self.cursor.fetchall()

        # 0 represents normal user
        if len(data) == 0:
            return 0
        # 1 represents spammer
        else:
            return 1

    def determine_spammer_by_percentage(self, reviewer_id):
        """
        determine the spammer by the percentage of fake reviews that a person publish
        :return:
        """
        cut_value = 0.5

        fake_sql = "select count(*) from reviews_simple where reviewerID = '%s' and fake = 1" % reviewer_id
        legitimate_sql = "select count(*) from reviews_simple where reviewerID = '%s' and fake = 0" % reviewer_id

        self.cursor.execute(fake_sql)
        fake_num = self.cursor.fetchone()
        self.cursor.execute(legitimate_sql)
        legitimate_num = self.cursor.fetchone()
        total_num = float(fake_num + legitimate_num)
        if total_num == 0:
            return 2  # 2 represents unknown label
        else:
            if fake_num/total_num > cut_value:
                return 1
            else:
                return 0

    def review_type(self, reviewer_id):
        """

        Parameters
        ----------
        reviewer_id

        Returns
        -------

        """
        sql_1 = "select count(*) from reviews_simple where reviewerID = '%s' and fake = 1" % reviewer_id
        sql_2 = "select count(*) from reviews_simple where reviewerID = '%s' and fake = 0" % reviewer_id

        self.cursor.execute(sql_1)
        number_of_normal_reviews = self.cursor.fetchone()
        number_of_fake_reviews = self.cursor.fetchone()
        return number_of_fake_reviews, number_of_normal_reviews

    def label_users(self):
        """
        label the user with fake, which means that whether a user is a spammer
        
        Returns
        -------

        """
        record_unit = 1000
        print self.friendship_graph.number_of_nodes()
        print self.friendship_graph.number_of_edges()

        for num, node in enumerate(self.friendship_graph.nodes()):
            fake_flag = self.determine_spammer(node)
            self.friendship_graph.node[node]['fake'] = fake_flag
            # print self.friendship_graph.node[node]
            if num % record_unit == 0:
                print num
                print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                nx.write_gpickle(self.friendship_graph, "graph/friendship_reviewer_label_attr%d.pickle" % num)
                if num != 0:
                    os.remove("graph/friendship_reviewer_label_attr%d.pickle" % (num - record_unit))

        nx.write_gpickle(self.friendship_graph, "graph/friendship_reviewer_label_attr%d.pickle" % num)

    def create_temporary_test(self):
        sql = "create temporary table reviewID_fake select id, reviewerID, fake from reviews "
        self.cursor.execute(sql)


if __name__ == "__main__":
    RFG = ReadFriendshipGraph()
    # flag = RFG.determine_spammer('DFWXDnzAyxQ7kxg12h4BlQ')
    # print flag
    RFG.label_users()
