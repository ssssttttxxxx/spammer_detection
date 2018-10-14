# -*- coding: utf-8 -*-

from sklearn import tree
import sklearn
print sklearn.__version__
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

result = clf.predict([[2, 2]])
print result