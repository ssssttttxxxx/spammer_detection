# -*- coding: utf-8 -*-

import MySQLdb
import networkx as nx
import time

class YelpData:

    def __init__(self):
        self.db = MySQLdb.connect("localhost", "root", "stx11stx11", "yelp_data", charset="utf8")
        self.cursor = self.db.cursor()

    def read_user_review(self, reviewerID):
        sql = "select * from reviews where reviewerID = '%s'" % reviewerID
        self.cursor.execute(sql)
        reviews_data = self.cursor.fetchall()
        return reviews_data

    def read_reviewer(self, reviewerID):
        sql = "select * from reviewers where id = '%s'" % reviewerID
        self.cursor.execute(sql)
        reviewer_data = self.cursor.fetchone()
        return reviewer_data


def number_of_reviews(review_data):
    return len(review_data)


def average_number_of_words(review_data):
    number_of_words = 0
    for record in review_data:
        number_of_words += len(record['ucontent'])

    return number_of_words/len(review_data)


def number_of_reviews_attitude():
    """
    add the attitude of the reviews to the graph
    Parameters
    ----------
    Returns
    -------

    """
    yd = YelpData()

    graph_path = 'graph/friendship_reviewer_label_attr_clean_unknown_degree0.pickle'
    graph = nx.read_gpickle(graph_path)
    print 'density', nx.density(graph)

    # data = yd.read_user_review('--F1tbZdtqrqx_Uh8rFFBw')
    # for d in data:
    #     print d
    review_data_list = list()
    for i, node_id in enumerate(graph.nodes()):

        review_origin_data = yd.read_user_review(node_id)
        num_pos_reviews = 0
        num_neu_reviews = 0
        num_neg_reviews = 0
        for data in review_origin_data:
            review_data_dict = dict()
            review_data_dict['reviewID'] = data[0]
            review_data_dict['businessId'] = data[1]
            review_data_dict['reviewerName'] = data[2]
            review_data_dict['reviewerID'] = data[3]
            review_data_dict['urating'] = data[4]
            review_data_dict['udate'] = data[5]
            review_data_dict['ucontent'] = data[6]
            review_data_dict['prating'] = data[7]
            review_data_dict['pdata'] = data[8]
            review_data_dict['pcontent'] = data[9]
            review_data_dict['pnum'] = data[10]
            review_data_dict['fake'] = data[11]
            review_data_list.append(review_data_dict)

            if int(review_data_dict['urating']) < 3:
                num_neg_reviews += 1
            elif int(review_data_dict['urating']) > 3:
                num_pos_reviews += 1
            elif int(review_data_dict['urating']) == 3:
                num_neu_reviews += 1
        if i % 100 == 0:
            print i

        graph.node[node_id]['pos_reviews'] = num_pos_reviews
        graph.node[node_id]['neg_reviews'] = num_neg_reviews
        graph.node[node_id]['neu_reviews'] = num_neu_reviews

    nx.write_gpickle(graph, 'graph/friendship_clean_attitude.pickle')


def number_of_reviews_one_day(review_data):
    pass


def ratio_of_pos_reviews(review_data):
    pass


def ratio_of_neg_reviews(review_data):
    pass


def second_degree_attr(review_data, graph, node_id):
    neighors_1 = nx.neighbors(graph, graph.node[node_id])
    for neighor in neighors_1:
        print neighor


def identify_same_location_user_type():
    yd = YelpData()

    graph_path = 'graph/firendship_new_label209579.pickle'
    graph = nx.read_gpickle(graph_path)

    for num, node in enumerate(graph.nodes()):

        if num % 1000 == 0:
            print num
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        reviewer_data = yd.read_reviewer(node)
        reviewer_location = reviewer_data[2]
        neighbors = graph.neighbors(node)

        same_location_spammer = 0
        same_location_legitimate = 0

        for neighbor in neighbors:
            neighbor_data = yd.read_reviewer(neighbor)
            neighbor_location = neighbor_data[2]
            if neighbor_location == reviewer_location:
                fake_flag = graph.node[neighbor]['fake']
                if fake_flag == 1:
                    same_location_spammer += 1
                elif fake_flag == 0:
                    same_location_legitimate += 1

        graph.node[node]['same_location_spammer_num'] = same_location_spammer
        graph.node[node]['same_location_legitimate_num'] = same_location_legitimate

    nx.write_gpickle(graph, 'graph/friendship_new_label_location.pickle')


if __name__ == "__main__":
    identify_same_location_user_type()

