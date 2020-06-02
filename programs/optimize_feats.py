#!/usr/bin/env python3
# Author: Tomer Gabay
# January - May 2020


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC


import pandas as pd
import argparse
import pickle

def identity(x):
    return x

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

def classify(feat,x_train_temp,y_train,dataset,settings_dict,df,i,j,name_vec,selector,analyz):

	""" Performs k-vold classification and checks if it outperforms the highest score per feature"""

	cls = LinearSVC()
	if name_vec == 'tfidf':
		vec = TfidfVectorizer(lowercase=False)
	else:
		vec = CountVectorizer(lowercase=False)
	pipe = Pipeline([
		('selector',TextSelector(key=selector)),
		('vec',vec)
	])

	pipe.set_params(vec__analyzer=analyz)
	pipe.set_params(vec__min_df=df)
	pipe.set_params(vec__ngram_range=(i,j))
	settings = {'Selector':selector,'cls':'LinearSVC','Vectorizer':name_vec,'Analyzer':analyz,'Min_df':df,'n-grams':(i,j)}
	
	if settings not in settings_dict[feat[0]+'_'+analyz[:4]]['tried']:
		settings_dict[feat[0]+'_'+analyz[:4]]['tried'].append(settings)
		try:
			pipe.fit(x_train_temp)
			classifier = Pipeline([
				('feats',pipe),
				('cls',cls)
			])
			scores = cross_validate(classifier, x_train_temp, y_train, cv=9, scoring=("accuracy","f1_macro"))
			f1 = round(sum(scores['test_f1_macro']) / len(scores['test_f1_macro']),3)
			if f1 > settings_dict[feat[0]+'_'+analyz[:4]]['max']['f1']:
				settings_dict[feat[0]+'_'+analyz[:4]]['max']['f1'] = f1
				settings_dict[feat[0]+'_'+analyz[:4]]['max']['settings'] = settings
		
		except:
			pass
		write_grid_search_settings_to_pickle(settings_dict,dataset)
	return settings_dict

def write_grid_search_settings_to_pickle(settings_dict, dataset):

	""" Writes settings to pickle file """

	with open("../datasets/settings/"+dataset+'_settings.p','wb') as outfile:
		pickle.dump(settings_dict,outfile)

def best_parameters_per_feature(feats,x_train,y_train,dataset):

	""" Manuel Grid search function """

	settings_dict = get_settings_dict(dataset,feats)
	for feat in feats:
		selector = feat[1]
		print("\n\n"+feat[0])
		for name_vec in ['tfidf']:
			print("vectorizer:",name_vec)
			for df in [5,10]:
				print("df:",df)
				for analyz in ['word','char_wb','char']:
					print("analyzer:",analyz)
					i_start = 1
					i_max = 10
					j_increment = 5
					if analyz == 'word':
						i_max = 4
						j_increment = 3
					for i in range(i_start,i_max,1):
						print("n-gram start:",i)
						for j in range(i,i + j_increment,1):
							settings_dict = classify(feat,x_train,y_train,dataset,settings_dict,df,i,j,name_vec,selector,analyz)
								
	return settings_dict

def return_feats():

	""" Links the feature to the appropriate column """

	feats = [
		('lexical_ngrams','lyrics'),
		('phonetic_ngrams','phonetic_repr'),
		('metaphone_ngrams','metaphone_repr'),
		('soundex_ngrams','soundex_repr'),
		('length_repr_ngrams','length_repr'),
		('punctC_repr_ngrams','punctC_repr'),
		('shape_repr_ngrams','shape_repr'),
		('vowel_repr_ngrams','vowel_repr'),
		('ner_repr_ngrams','ner_repr'),
		('pos_repr_ngrams','pos_repr'),
		('syllable_repr_ngrams','syllab_repr'),
	]
	return feats

def parse():

	""" Set up the argparse """

	parser = argparse.ArgumentParser(\
		description='Optimizing the feature settings and writing it to a file. Select data set with -ds argument')
	parser.add_argument('-ds','--dataset', type=str, metavar='', \
		help="choose which data set to use: 'DADS', 'AAMDS','GBDS', 'CADSa' or 'CADSb", default='DADS')
	return parser.parse_args()

def get_settings_dict(dataset,feats):

	""" Make sure to have a settings_dict file by trying to load it and if that fails create one """

	try:
		with open("../datasets/settings/"+dataset+"_settings.p",'rb') as pickle_in:
			settings_dict = pickle.load(pickle_in)
			print("Succesfully loaded previous optimized settings")
	except:
		settings_dict = {}
		for feat in feats:
			settings_dict[feat[0]+'_char'] = {'max':{'f1':0,'settings':''},'tried':[]}
			settings_dict[feat[0]+'_word'] = {'max':{'f1':0,'settings':''},'tried':[]}
		write_grid_search_settings_to_pickle(settings_dict,dataset)
	return settings_dict


def main():
	print("Searching for the optimal settings per feature. This can take several hours.")
	args = parse()
	feats = return_feats()
	x_train = pd.read_csv("../datasets/"+args.dataset+"_train.csv")
	x_train = x_train[x_train.type != 'verses'] # remove verses
	y_train = x_train['artist']
	best_parameters_per_feature(feats,x_train,y_train,args.dataset)
	print("Succesfully stored the optimal settings per feature in ../datasets/settings/"+args.dataset+"_tried_settings.p")

if __name__ == '__main__':
	main()