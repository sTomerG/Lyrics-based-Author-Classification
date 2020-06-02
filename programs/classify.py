#!/usr/bin/env python3
# classify.py
# Author: Tomer Gabay
# January - May 2020

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

import warnings
import argparse
import pickle

warnings.filterwarnings('ignore')

class TextSelector(BaseEstimator, TransformerMixin): # https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
	
	"""Use on text columns in the data"""
	
	def __init__(self, key):
		self.key = key

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin): # https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
	
	"""Use on numeric columns in the data"""

	def __init__(self, key):
		self.key = key

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[[self.key]]



def str2bool(v): # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
	
	""" Convert inputted string to bool """

	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():

	""" Set up the argparse """

	parser = argparse.ArgumentParser(\
		description='Classify author attribution on lyrics. Select data set with -ds argument')
	parser.add_argument('-ds','--dataset', type=str, metavar='', \
		help="choose which data set to use: 'DADS', 'AAMDS','GBDS', 'CADSa' or 'CADSb", required=True)
	parser.add_argument('-v','--verses', type=str2bool, metavar='', \
		help="Use verses in the training set, True means yes, False means no. Default is true", default=True)
	parser.add_argument('-c', '--c_value', type=float, metavar='', \
		help="Set the C value to the entered number. Default is 1, 200 is suggested.", default=1)
	parser.add_argument('-t', '--test', type=str, metavar='', \
		help="choose which data set to use as test data: 'dev', 'test' or 'kvold'. Default is 'test'",default='test')
	return parser.parse_args()

def get_x_and_y(args):

	""" Generates the train and test set based on the argparse """

	x_train = pd.read_csv("../datasets/"+args.dataset+"_train.csv")
	if args.verses == False:
		x_train = x_train[x_train.type != 'verses']
	
	if args.test != "kvold":
		x_test = pd.read_csv("../datasets/"+args.dataset+"_"+args.test+".csv")
		y_test = x_test['artist']
		kvold = False
	else:
		x_test = ''
		y_test = ''
		kvold = True
		x_train = x_train[x_train.type != 'verses']

	y_train = x_train['artist']
	return x_train,y_train,x_test,y_test, kvold

	

def get_results(x_train,y_train,x_test,y_test,feat_set,c,kvold):

	""" Does the actual classification """

	feats = FeatureUnion(feat_set)
	feature_processing = Pipeline([('feats', feats)])
	feature_processing.fit_transform(x_train)

	cls = LinearSVC(C=c)    #cls = RandomForestClassifier()
	classifier = Pipeline([
		('feats', feats),
		('cls', cls)
	])
	if kvold == True:
		scores = cross_validate(classifier, x_train, y_train, cv=9, scoring=("accuracy","f1_macro"))
		print("Average macro f1 score with K-Vold (k=9): {0}\n ".format\
			  (round(sum(scores['test_f1_macro']) / len(scores['test_f1_macro']),3)))
		print("Average accuracy with K-Vold (k=9): {0}\n ".format\
			  (round(sum(scores['test_accuracy']) / len(scores['test_accuracy']),3)))
	else:
		classifier.fit(x_train,y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy: {}, Macro-F1: {}".format(round(accuracy_score(y_test,y_pred),3),
												  round(f1_score(y_test,y_pred,average="macro"),3)))
		print("\n"+classification_report(y_test,y_pred))
		class_labels = classifier.classes_
		disp = plot_confusion_matrix(classifier, x_test, y_test,
                             display_labels=[c[:4] for c in class_labels],
                             cmap=plt.cm.Blues)
		conf_matrix = confusion_matrix(y_test, y_pred)
		plt.show()


def main():
	args = parse()
	print("Classifying.. this might take several minutes.\n")
	with open("../datasets/settings/"+args.dataset+"_best_feat_set.p",'rb') as pickle_in:
		feat_set = pickle.load(pickle_in)
	feat_set = [(feat_name,pipe) for feat_name,pipe in feat_set.items()]
	x_train,y_train,x_test,y_test, kvold = get_x_and_y(args)
	get_results(x_train,y_train,x_test,y_test,feat_set,args.c_value,kvold)



if __name__ == '__main__':
	main()