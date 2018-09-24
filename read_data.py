# -*- coding: utf-8 -*-

import MySQLdb
import networkx as nx
# import matplotlib.pyplot as plt
import os
import time

class YelpData:

    def __init__(self):
        self.db = MySQLdb.connect("quantum2.is.cityu.edu.hk.", "readyelp", "yelp2018", "yelp", charset="utf8")
        self.cursor = self.db.cursor()

    def read_reviews(self):
        reviews_sql = "select * from reviews limit 10"
        self.cursor.execute(reviews_sql)
        data = self.cursor.fetchall()
        # for single_data in data:
        #     print single_data
        return data

    def read_reviewer(self):
        reviewer_sql = "select * from reviewer limit 10"
        self.cursor.execute(reviewer_sql)
        data = self.cursor.fetchall()
        for single_data in data:
            print single_data

    def read_bussiness(self):
        bussiness_sql = "select * from bussiness limit 10"
        self.cursor.execute(bussiness_sql)
        data = self.cursor.fetchall()
        for single_data in data:
            print single_data

    def read_reviews_limit(self, num, limit):
        reviews_sql = "select * from reviews limit %d, %d" % (num, limit)
        print reviews_sql
        self.cursor.execute(reviews_sql)
        data = self.cursor.fetchall()
        # for single_data in data:
        #     print single_data
        return data

    def construct_network_test(self):
        # user-review-product
        graph_urp = nx.Graph()
        reviews = self.read_reviews()
        for reviews_data in reviews:
            # print reviews_data
            review_id = reviews_data[0]
            bussiness_id = reviews_data[1]
            reviewer_name = reviews_data[2]
            reviewer_id = reviews_data[3]
            urating = reviews_data[4]
            udate = reviews_data[5]
            ucontent = reviews_data[6]
            prating = reviews_data[7]
            pdate = reviews_data[8]
            pcontent = reviews_data[9]
            pnum = reviews_data[10]
            fake = reviews_data[11]
            graph_urp.add_node(review_id,ucontent=ucontent, urating=urating, fake=fake)
            graph_urp.add_node(reviewer_id, reviewer_name=reviewer_name, udate=udate)
            graph_urp.add_node(bussiness_id, prating=prating, pdate=pdate, pcontent=pcontent, pnum=pnum)
            graph_urp.add_edge(review_id, reviewer_id)
            graph_urp.add_edge(review_id, bussiness_id)

        # nx.draw(graph_urp)
        # plt.savefig('pic/test_graph.png')
        # plt.show()
        # graph_urp.add_node()

    def construct_network_urp(self):
        # user-review-product network


        # define limit, offset, rest
        row_sum = 10876672
        # row_sum = 10
        read_unit = 1000
        rest_sum = row_sum-row_sum/read_unit*read_unit

        # if no record point, create a new graph
        # graph_urp = nx.Graph()

        # read pickle if need to recover, otherwise comment the next 2 lines
        record_point = 622
        graph_urp = nx.read_gpickle("graph/test_%d.pickle" % (record_point*1000))

        for i in range(record_point, row_sum/read_unit+1):
        # for i in range(1, 10):

            if i == row_sum/read_unit:
                print 10876672
                reviews = self.read_reviews_limit(i*read_unit, rest_sum)
            else:
                print (i+1)*read_unit
                reviews = self.read_reviews_limit(i*read_unit, read_unit)
                # reviews = self.read_reviews_limit(i*10, 10)

            for reviews_data in reviews:
                # print reviews_data
                review_id = reviews_data[0]
                bussiness_id = reviews_data[1]
                reviewer_name = reviews_data[2]
                reviewer_id = reviews_data[3]
                urating = reviews_data[4]
                udate = reviews_data[5]
                ucontent = reviews_data[6]
                prating = reviews_data[7]
                pdate = reviews_data[8]
                pcontent = reviews_data[9]
                pnum = reviews_data[10]
                fake = reviews_data[11]
                graph_urp.add_node(review_id, type="review", ucontent=ucontent, urating=urating, fake=fake)
                graph_urp.add_node(reviewer_id, type="reviewer", reviewer_name=reviewer_name, udate=udate)
                graph_urp.add_node(bussiness_id, type="bussiness", prating=prating, pdate=pdate, pcontent=pcontent, pnum=pnum)
                graph_urp.add_edge(review_id, reviewer_id)
                graph_urp.add_edge(review_id, bussiness_id)

            # # write adjlist
            # nx.write_adjlist(graph_urp, "graph/test1.adjlist")
            # # write edgelist
            # nx.write_edgelist(graph_urp, "graph/test1.edgelist")
            # # write GEXF
            # nx.write_gexf(graph_urp, "graph/test1.gexf")

            # write pickle
            nx.write_gpickle(graph_urp, "graph/test_%d.pickle" % ((i+1)*read_unit))
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # delete the old pickle
            if i != 0:
                os.remove("graph/test_%s.pickle" % str(i*read_unit))

        # nx.draw(graph_urp)
        # plt.savefig('pic/test_graph.png')
        # plt.show()


    def construct_network_friendship(self):
        pass


    def read_graph(self, path):
        graph = nx.read_gpickle(path)
        return graph
        # graph = nx.read_gexf(path)
        # return graph
if __name__ == "__main__":
    YD = YelpData()
    YD.construct_network_urp()

    # graph = YD.read_graph("graph/test1.pickle")
    # for node in graph.nodes():
    #     print graph.node[node]
