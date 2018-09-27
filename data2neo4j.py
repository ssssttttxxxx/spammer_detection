# -*- coding: utf-8 -*-
from neo4j.v1 import GraphDatabase
import chardet
import time
from read_data import YelpData
from py2neo import Node, Relationship, Graph


import sys
reload(sys)
sys.setdefaultencoding('utf8')

uri = "http://localhost:7474/"

class Neo4jGraph():
    def __init__(self):
        self.neo4j_graph = Graph(
            "http://localhost:7474/",
            username='neo4j',
            password='123456'
        )
    def insert(self):
        YD = YelpData()

        row_sum = 10876672
        # row_sum = 10
        read_unit = 100
        rest_sum = row_sum - row_sum / read_unit * read_unit
        record_point = 2971

        for i in range(record_point, row_sum/read_unit+1):
            # reviews = YD.read_reviews_limit(1, 10)
            self.tx = self.neo4j_graph.begin()

            if i == row_sum/read_unit:
                print 10876672
                reviews = YD.read_reviews_limit(i*read_unit, rest_sum)
            else:
                print (i+1)*read_unit
                reviews = YD.read_reviews_limit(i*read_unit, read_unit)
                # reviews = self.read_reviews_limit(i*10, 10)

            for reviews_data in reviews:
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


                # # 编码处理
                # ucontent_encoding = chardet.detect(ucontent)['encoding']
                # # print ucontent_encoding
                # ucontent_decoded = ucontent.decode(ucontent_encoding).encode('utf8')
                # # print chardet.detect(ucontent_decoded)
                # pcontent_encoding = chardet.detect(pcontent)['encoding']
                # # print pcontent_encoding
                # pcontent_decoded = pcontent.decode(pcontent_encoding).encode('utf8')
                # # print chardet.detect(pcontent_decoded)
                # if reviewer_name is not "":  # 名字可能为空
                #     reviewer_name_encoding = chardet.detect(reviewer_name)['encoding']
                #     reviewer_name_decoded = reviewer_name.decode(reviewer_name_encoding).encode('utf8')
                # else:
                #     reviewer_name_decoded = ""


                # construct relationship
                reviewer_node = Node('reviewer', id=reviewer_id,
                                     reviewer_name=reviewer_name, udate=udate)
                self.tx.merge(reviewer_node, 'reviewer', "id")

                review_node = Node('review', id=review_id,
                                   ucontent=ucontent, urating=urating, fake=fake)
                self.tx.merge(review_node, primary_label='review', primary_key="id")

                product_node = Node('product', id=bussiness_id,
                                    prating=prating, pdate=pdate, pcontent=pcontent, pnum=pnum)
                self.tx.merge(product_node, primary_label='product', primary_key="id")

                reviewer_review = Relationship(reviewer_node, 'publish', review_node)
                review_product = Relationship(review_node, 'comment', product_node)

                # if self.tx.exists(reviewer_review):
                #     print "not exists rr"
                self.tx.merge(reviewer_review)
                # if self.tx.exists(review_product):
                #     print "not exists rp"
                self.tx.merge(review_product)
                # self.tx.push(reviewer_review)
                # self.tx.push(review_product)
            print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            self.tx.commit()

    def delete_all(self):
        self.neo4j_graph.delete_all()



if __name__ == "__main__":
    NG = Neo4jGraph()
    NG.insert()
    # NG.delete_all()
