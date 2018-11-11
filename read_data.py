# -*- coding: utf-8 -*-

import MySQLdb
import networkx as nx
# import matplotlib.pyplot as plt
import os
import time

class YelpData:

    def __init__(self):
        self.db = MySQLdb.connect("server_address", "username", "password", "yelp", charset="utf8")

        self.cursor = self.db.cursor()
        self.mydb = MySQLdb.connect("localhost", "root", "stx11stx11", "yelp_data", charset="utf8")
        self.mycursor = self.mydb.cursor()

    def read_reviews(self):
        reviews_sql = "select * from reviews limit 10"
        self.cursor.execute(reviews_sql)
        data = self.cursor.fetchall()
        # for single_data in data:
        #     print single_data
        return data

    def read_reviewer(self):
        reviewer_sql = "select * from reviewers limit 10"
        self.cursor.execute(reviewer_sql)

        data = self.cursor.fetchall()
        for single_data in data:
            print single_data

    def read_reviewer_single(self, reviewerID):
        reviewer_sql = "select * from reviewers where id = '%s'" % reviewerID
        self.cursor.execute(reviewer_sql)
        data = self. cursor.fetchone()
        # print data
        return data

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
        return data

    def read_friends_limit(self, num, limit):
        friends_sql = "select * from friends order by id desc limit %d, %d" % (num, limit)
        print friends_sql
        self.cursor.execute(friends_sql)
        data = self.cursor.fetchall()
        return data

    def determine_spammer(self, reviewer_id):
        # whether the reviewer is a spammer
        fake_sql = "select * from reviews_simple where reviewerID = '%s' and fake = 1" % reviewer_id
        self.mycursor.execute(fake_sql)
        data = self.mycursor.fetchall()
        if len(data) == 0:
            return False
        else:
            return True

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
        graph_urp = nx.read_gpickle("graph/test_%d.pickle" % (record_point * read_unit))

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
            nx.write_gpickle(graph_urp, "graph/friendship_%d.pickle" % ((i+1)*read_unit))
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # delete the old pickle
            if i != 0:
                os.remove("graph/test_%s.pickle" % str(i*read_unit))

        # nx.draw(graph_urp)
        # plt.savefig('pic/test_graph.png')
        # plt.show()

    def construct_network_friendship(self):
        # define limit, offset, rest
        row_sum = 925091
        read_unit = 100
        rest_sum = row_sum - row_sum / read_unit * read_unit



        record_point = 0
        if record_point != 0:
            # read pickle if need to recover, otherwise comment the next 2 lines
            graph_friendship = nx.read_gpickle("graph/friendship_attr_order_%d.pickle" % (record_point * read_unit))
        else:
            # if no record point, create a new graph
            graph_friendship = nx.Graph()

        for i in range(record_point, row_sum / read_unit + 1):
            # for i in range(1, 10):

            if i == row_sum / read_unit:
                print 925091
                friendships = self.read_friends_limit(i * read_unit, rest_sum)
            else:
                print (i + 1) * read_unit
                friendships = self.read_friends_limit(i * read_unit, read_unit)

            for friendship in friendships:
                reviewerID_0 = friendship[0]
                reviewerID_1 = friendship[1]

                # with attributes of reviewer
                reviewer_detail_0 = self.read_reviewer_single(reviewerID_0)
                reviewer_detail_1 = self.read_reviewer_single(reviewerID_1)
                if reviewer_detail_0 is not None:
                    graph_friendship.add_node(
                        reviewerID_0, reviewerID=reviewerID_0,
                        reviewerName=reviewer_detail_0[1],
                        location=reviewer_detail_0[2],
                        friends_num=reviewer_detail_0[3],
                        reviews_num=reviewer_detail_0[4],
                        photo_num=reviewer_detail_0[5]
                    )
                else:
                    graph_friendship.add_node(
                        reviewerID_0,
                        reviewerID=reviewerID_0,
                        reviewerName='unknown',
                        location='unknown',
                        friends_num='unknown',
                        reviews_num='unknown',
                        photo_num='unknown'
                    )
                if reviewer_detail_1 is not None:
                    graph_friendship.add_node(
                        reviewerID_1, reviewerID=reviewerID_1,
                        reviewerName=reviewer_detail_1[1],
                        location=reviewer_detail_1[2],
                        friends_num=reviewer_detail_1[3],
                        reviews_num=reviewer_detail_1[4],
                        photo_num=reviewer_detail_1[5]
                    )
                else:
                    graph_friendship.add_node(
                        reviewerID_1,
                        reviewerID=reviewerID_1,
                        reviewerName='unknown',
                        location='unknown',
                        friends_num='unknown',
                        reviews_num='unknown',
                        photo_num='unknown'
                    )

                # with fake label, but slow
                # fake_0 = self.determine_spammer(reviewerID_0)
                # fake_1 = self.determine_spammer(reviewerID_1)
                # graph_friendship.add_node(reviewerID_0, reviewerID=reviewerID_0, spammer=fake_0)
                # graph_friendship.add_node(reviewerID_1, reviewerID=reviewerID_1, spammer=fake_1)

                # without fake lable
                # graph_friendship.add_node(reviewerID_0, reviewerID=reviewerID_0)
                # graph_friendship.add_node(reviewerID_1, reviewerID=reviewerID_1)
                graph_friendship.add_edge(reviewerID_0, reviewerID_1)

            nx.write_gpickle(graph_friendship, "graph/friendship_attr_order_%d.pickle" % ((i + 1) * read_unit))
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # delete the old pickle
            if i != 0:
                os.remove("graph/friendship_attr_order_%s.pickle" % str(i * read_unit))

    def construct_network_spammer(self):
        # define limit, offset, rest
        num = 0

        row_sum = 925091
        read_unit = 100
        rest_sum = row_sum - row_sum / read_unit * read_unit

        record_point = 0
        if record_point != 0:
            # read pickle if need to recover, otherwise comment the next 2 lines
            graph_spammer = nx.read_gpickle("graph/spammer_%d.pickle" % (record_point * read_unit))
        else:
            # if no record point, create a new graph
            graph_spammer = nx.Graph()

        for i in range(record_point, row_sum / read_unit + 1):
            # for i in range(1, 10):

            if i == row_sum / read_unit:
                print 925091
                friendships = self.read_friends_limit(i * read_unit, rest_sum)
            else:
                print (i + 1) * read_unit
                friendships = self.read_friends_limit(i * read_unit, read_unit)

            for friendship in friendships:
                reviewerID_0 = friendship[0]
                reviewerID_1 = friendship[1]
                if self.determine_spammer(reviewerID_0) and self.determine_spammer(reviewerID_1):
                    graph_spammer.add_node(reviewerID_0, reviewerID=reviewerID_0, spammer=1)
                    graph_spammer.add_node(reviewerID_1, reviewerID=reviewerID_1, spammer=1)
                    graph_spammer.add_edge(reviewerID_0, reviewerID_1)
                    num += 1

            if num % 100 == 0:
                print 'spammer', num
                print 'spammer', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            nx.write_gpickle(graph_spammer, "graph/spammer_%d.pickle" % ((i + 1) * read_unit))
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # delete the old pickle
            if i != 0:
                os.remove("graph/spammer_%s.pickle" % str(i * read_unit))

    def read_graph(self, path):
        graph = nx.read_gpickle(path)
        return graph
        # graph = nx.read_gexf(path)
        # return graph


if __name__ == "__main__":
    YD = YelpData()
    # YD.construct_network_urp()
    # YD.determine_spammer('wwjSNPlPDONJ7sOAwXkftA')
    # YD.read_reviewer_single('I9_Eevm2Be99teo5v9qNwg')
    # YD.construct_network_friendship()
    YD.construct_network_spammer()
    # graph = YD.read_graph("graph/test1.pickle")
    # for node in graph.nodes():
    #     print graph.node[node]
