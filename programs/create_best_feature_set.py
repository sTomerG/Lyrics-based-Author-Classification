#!/usr/bin/env python3
# create_best_feature_set.py
# Author: Tomer Gabay
# January - May 2020

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

import pandas as pd

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
		description='Optimizing the feature settings and writing it to a file. Select data set with -ds argument')
	parser.add_argument('-ds','--dataset', type=str, metavar='', \
		help="choose which data set to use: 'DADS', 'AAMDS','GBDS', 'CADSa' or 'CADSb'", required=True)
	parser.add_argument('-ow', '--overwrite', type=str2bool,metavar='', \
		help="overwrite the previoulsy determined outcome, choose true or false. Default is false", default=False)
	return parser.parse_args()


def try_feats_combinations(settings_dict,x_train,y_train):

	""" Add each feature to the active feature set to determine whether a feature contributes """

	best_feats_set = {}
	max_f1 = 0
	active_feats = []
	for key in sorted(settings_dict.keys(),reverse=True):
		if settings_dict[key]['settings']['cls'] == "LinearSVC":
			cls = LinearSVC()
		if settings_dict[key]['settings']['Vectorizer']== 'tfidf':
			vec = TfidfVectorizer(lowercase=False)
		elif settings_dict[key]['settings']['Vectorizer']== 'count':
			vec = CountVectorizer(lowercase=False)
		pipe = Pipeline([
			('selector',TextSelector(key=settings_dict[key]['settings']['Selector'])),
			('vec',vec)
		])
		pipe.set_params(vec__analyzer=settings_dict[key]['settings']['Analyzer'])
		pipe.set_params(vec__min_df=settings_dict[key]['settings']['Min_df'])
		pipe.set_params(vec__ngram_range=settings_dict[key]['settings']['n-grams'])
		
		active_feats.append((settings_dict[key]['feature'],pipe))
		featsUnioned = FeatureUnion(active_feats)
		feature_processing = Pipeline([('feats', FeatureUnion(active_feats))])
		feature_processing.fit_transform(x_train)
		classifier = Pipeline([
		('feats', featsUnioned),
		('cls', cls)])
		x_train_temp = x_train.copy()
		classifier.fit(x_train_temp, y_train)
		scores = cross_validate(classifier, x_train_temp, y_train, cv=9, scoring=("accuracy","f1_macro"))
		f1 = round(sum(scores['test_f1_macro']) / len(scores['test_f1_macro']),3)

		if f1 > max_f1:
			best_feats_set[settings_dict[key]['feature']] = pipe
			max_f1 = f1
		else:
			active_feats = active_feats[:-1]
	return best_feats_set


def create_best_settings_dict(settings_dict):

	""" Extract the best settings from the settings dict """

	best_settings_dict = {}
	for key,value in settings_dict.items():
		best_settings_dict[settings_dict[key]['max']['f1']] = {'feature':key,'settings':settings_dict[key]['max']['settings']}
	return best_settings_dict

def check_if_should_run(args):

	""" check if the program should run based on the previous outcome and the overwrite argument """

	if args.overwrite == False:
		try:
			with open("../datasets/settings/"+args.dataset+"_best_feats.p",'rb') as pickle_in:
				best_feats = pickle.load(pickle_in)
			print("Outcome already stored in: ../datasets/settings/"+args.dataset+"_best_feats.p")
			print("To rerun and overwrite, use -ow True. This could take a few hours.")
			return False
		except:
			pass
	print("Determining the best feature set. This could take a few hours.")
	return True

def main():
	args = parse()
	run = check_if_should_run(args)
	if run == True:
		x_train = pd.read_csv("../datasets/"+args.dataset+"_train.csv")
		x_train = x_train[x_train.type != 'verses'] # remove verses
		y_train = x_train['artist']
		with open("../datasets/settings/"+args.dataset+"_tried_settings.p",'rb') as pickle_in:
			settings_dict = pickle.load(pickle_in)
		best_settings_dict = create_best_settings_dict(settings_dict)
		best_feats_set = try_feats_combinations(best_settings_dict,x_train,y_train)
		with open("../datasets/settings/"+args.dataset+"_best_feat_set.p",'wb') as pickle_out:
			pickle.dump(best_feats_set, pickle_out)


	



if __name__ == '__main__':
	main()