# -*- coding: utf-8 -*-
import pickle
model_path = 'decision_tree_result/decision_tree_model43.sav'
clf = pickle.load(open(model_path, 'rb'))
print clf.predict([[0,9,3,1,0]])