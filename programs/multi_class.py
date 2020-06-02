import pickle
import os
import random
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import sys
import re
from itertools import combinations
import argparse
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
import nltk
import spacy
from spacy import displacy
import en_core_web_sm
import matplotlib.pyplot as plt


def import_data(path):
	""" Reads in the pickle files with lyrics and returns x (lyrics) and y (labels/artists) """

	# read in the pickle files per artist and append them to one list
	artist_dict_list = []
	for filename in os.listdir(path):
		if str(filename)[-2:] == '.p':
			with open(path+filename, "rb") as f:
				artist_dict = pickle.load(f)	
			artist_dict_list.append(artist_dict)

	# create a list of tuples with (lyrics, artist)
	all_songs = []
	for artist_dict in artist_dict_list:
		for song_title, song_info in artist_dict.items():
			all_songs.append((song_info[3], song_info[0])) #song_info[3] = lyrics, song_info[0] = artist
	# form x and y	
	x, y = shuffe_x_and_y(all_songs)	
	c = Counter(y)
	amount_of_songs = [c[artist] for artist in c]
	amount_of_songs.sort()
	print(amount_of_songs)
	x = preprocess_lyrics(x,y)
	return x,y

def balance_afro_male_data_to_diverse():
	""" Balances afro male data set to have same amount of songs per artist as the diverse artist set """

	x_afro_male, y_afro_male = import_data(convert_to_path("afro_male"))
	x_diverse, y_diverse = import_data(convert_to_path("diverse"))

	c_diverse = Counter(y_diverse).most_common()
	c_afro_male = Counter(y_afro_male).most_common()

	afro_male_dict = defaultdict(list)
	new_x, new_y = [], []
	for song, artist in zip(x_afro_male, y_afro_male): 
		afro_male_dict[artist].append(song) 

	for diverse, afro_male in zip(c_diverse, c_afro_male):
		songs_diverse_artist = diverse[1]
		afro_male_artist = afro_male[0]
		new_x += afro_male_dict[afro_male_artist][:songs_diverse_artist]
		new_y += [afro_male_artist for i in range(songs_diverse_artist)]
	x, y = shuffe_x_and_y([(x,y) for x,y in zip(new_x,new_y)])
	x = preprocess_lyrics(x,y)
	return x, y


def shuffe_x_and_y(x_and_y):
	""" Returns a shuffled x and y from a list with [(x,y), (x,y)...]"""
	random.seed(30)
	random.shuffle(x_and_y) # shuffle to have a different train and test set for each iteration
	new_x, new_y = [], []
	for x,y in x_and_y:
		new_x.append(x)
		new_y.append(y)
	return new_x, new_y

def get_nicknames():
	""" Returns a dictionairy of nicknames per artist, based on Wikipedia """

	return {"JAY-Z":["Jay","Hova","HOV","hov","Hov","Jigga","Shawn Carter","Shawn","Carter"],
	"Eminem": ["Marshall Mathers","Marshall","Mathers","Slim Shady","Slim","Shady"],
	"Future": ["Nayvadius Wilburn","Neyvadius","Wiburn","Meathead","Caeser Lee","Ceaser","Lee"],
	"Ice Cube": ["Ice","Cube","O'Shea Jackson","O'Shea","Jackson"],
	"Lil’ Kim": ["Lil’","Kim","Kimberley Jones","Kimberley","Jones","Queen Bee","Queen","Bee","Lil'","own_nameme", "own_name own_name"],
	"Machine Gun Kelly": ["Machine Gun","Gun Kelly","Kelly","Kells","Richard Baker","Richard","Baker"],
	"Nas": ["own_namety","Escobar", "Jones"],
	"Nicki Minaj": ["Nicki","Minaj","Onika Maraj","Onika","Maraj"],
	"50 Cent": ["50","Cent","Ferrari F-50","Ferrari","F-50","Curtis Jackson","Curtis","Jackson"],
	"2Pac": ["Tupac Shakur","Tupac","Shakur","Makaveli","MC New York"],
	"Lil Wayne": ["Wayne","Tunechi","Weezy F. Baby", "Weezy","President Carter","Dwayne Carter","Dwayne","Carter"],
	"Snoop Dogg": ["Snoop","Dogg","DJ Snoopadelic","Snoopadelic","Niggarachi","Snoopzilla","Nemo Hoes","Nemo"]
	}

def preprocess_lyrics(x,y):
	nicknames = get_nicknames()
	filtered_x = []
	for song,artist in zip(x,y):
		#print(artist)
		song = re.sub("\[.*\]", "", song)
		song = re.sub(artist,"own_name",song)
		for nickname in nicknames[artist]:
			song = re.sub(nickname,"own_name",song)
		#song = song.lower()
		song = re.sub("\n+"," . ", song)
		filtered_x.append(" ".join(nltk.word_tokenize(song)))
	#print(filtered_x[0])
	return filtered_x

def generate_feats():
	char_ngrams = ('char_ngrams', TfidfVectorizer(preprocessor=identity, tokenizer=tokenize, ngram_range=(3,6), analyzer='char_wb'))
	word_count = ('word_count'), Pipeline([('word_count_feat',FunctionTransformer(get_word_count, validate=False))])
	exclam_mark_count = ('exclam_mark_count', Pipeline([('exclam_mark_feat', FunctionTransformer(get_exclam_mark_count, validate=False))])) 
	unique_words = ('unique_words', Pipeline([('uniq_word_feat', FunctionTransformer(get_unique_words_ratio, validate=False))]))
	avg_word_length = ('avg_word_length', Pipeline([('avg_word_length_feat', FunctionTransformer(get_avg_word_length, validate=False))]))
	question_mark_count = ('question_mark_count',Pipeline([('question_mark_count_feat',FunctionTransformer(get_question_mark_count, validate=False))]))
	word_ngrams = ('word_ngrams', TfidfVectorizer(preprocessor=identity, ngram_range=(1,3)))
	#question_mark_count = ('question_mark_count', TfidfVectorizer(preprocessor=get_question_mark_count))
	#avg_sentence_length = ('avg_sentence_length'), Pipeline([('avg_sentence_length_feat',FunctionTransformer(get_avg_sentence_length, validate=False))])
	print(question_mark_count)
	return [
	#char_ngrams,
	#word_count,
	#exclam_mark_count,
	#unique_words,
	avg_word_length,
	#question_mark_count,
	#word_ngrams,
	#avg_sentence_length
	]

def identity(x):
	"""A dummy function that just returns its input"""

	return x

def get_NER_tags(x):
	nlp = en_core_web_sm.load()
	new_x = []
	for lyrics in x:
		ner_tag_list = []
		sentences = lyrics.split(' . ')
		for sentence in sentences:
			print(sentence)
			ner_tags = nlp(sentence)
			print(ner.label_ for ner in ner_tags.ents)
			ner_tag_list.append(ner.label_ for ner in ner_tags.ents)
		new_x.append(" ".join(ner_tag_list))
	print(new_x)


"""def get_avg_sentence_length(x):

	return [sum(len(s.split())) for s in l.split(' . ') / len(l.split(' . ')) for l in x]"""

	
def tokenize(x):
	return nltk.word_tokenize(x)

def get_word_count(x):
	return np.array([len(lyrics.split()) for lyrics in x]).reshape(-1,1)

def get_exclam_mark_count(x):
	return np.array([lyrics.count('!') for lyrics in x]).reshape(-1,1)

def get_unique_words_ratio(x):
	return np.array([len(set(lyrics))/len(lyrics.split()) for lyrics in x]).reshape(-1,1)

def get_avg_word_length(x):
	#print(np.array([sum(len(word) for word in lyrics.split()) / len(lyrics.split()) for lyrics in x]).reshape(-1,1))
	return np.array([sum(len(word) for word in lyrics.split()) / len(lyrics.split()) for lyrics in x]).reshape(-1,1)

def get_question_mark_count(x):
	#print(len(x))
	y= np.array([lyrics.count('?') for lyrics in x]).reshape(-1,1)
	#print(y[:3])
	return np.array([lyrics.count('?') for lyrics in x]).reshape(-1,1)

#def get_POS_tags(x):



def try_vis(conf_matrix, y_test, y_pred):
	""" Try to visualize confusion matrix """
	try:
		vis(conf_matrix, sorted(list(set(y_test))))
	except:
		try:
			vis(conf_matrix, sorted(list(set(y_pred))))
		except:
			pass

def vis(conf_mat, labels):
	'''Visualise the Confusion matrix'''

	df_cm = pd.DataFrame(conf_mat, index=[i for i in labels], columns=[i for i in labels])
	plt.figure(figsize=(10,7))
	plt.title('Confusion matrix')
	sn.heatmap(df_cm, annot=True, fmt='g')
	plt.show()

def split_train_test(x,y,boundary):
	return x[:boundary], y[:boundary], x[boundary:], y[boundary:]

def convert_to_gender_classification(x,y):
	""" Converts classification task to gender """

	new_x, new_y = [], []
	total_female_songs = y.count("Nicki Minaj") + y.count("Lil’ Kim")
	new_x_and_y = []
	total_male_songs = 0
	for song,artist in zip(x,y):
		if artist == "Lil’ Kim" or artist == "Nicki Minaj":
			new_x_and_y.append((song,'female'))
		elif total_male_songs < total_female_songs:
			total_male_songs += 1
			new_x_and_y.append((song,'male'))
	random.shuffle(new_x_and_y)
	new_x = [song for song,gender in new_x_and_y]
	new_y = [gender for song,gender in new_x_and_y]

	return new_x, new_y

def most_informative_features(x,y):
	""" Gives a list of the coefficient per word, all of them, and gives  """
	boundary = int(0.8*len(x))
	x_train, y_train, x_test, y_test = split_train_test(x,y,boundary)


	vec = TfidfVectorizer(ngram_range=(3, 6), analyzer='char_wb')
	x_train = vec.fit_transform(x_train)
	
	classifier = LinearSVC(C=200)
	classifier.fit(x_train, y_train)
	
	#classifier.fit(x_train, y_train)
	#classifier.score(x, y)

	feature_names = vec.get_feature_names()
	coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))

	df = pd.DataFrame(coefs_with_fns)
	df.columns = 'coefficient', 'word'
	df.sort_values(by='coefficient')
	# print(df) # to print all coefficients

	plot_coefficients(classifier, feature_names)

def plot_coefficients(classifier, feature_names, top_features=20):
	"""To plot the top X features"""
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

	# create plot
	plt.figure(figsize=(17, 15))
	#plt.title('Top {0} features for {1}'.format(top_features))
	plt.xlabel('Feature')
	plt.ylabel('Coefficient')
	plt.ylim(-6, 6)
	colors = []
	for c in coef[top_coefficients]:
		if c < 0:
			colors.append('green')
		else:
			colors.append('red')

	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)

	fet_list = (feature_names[top_coefficients].tolist()) # for when you want translation
	fet_names = feature_names[top_coefficients]

	plt.xticks(np.arange(1, 1 + 2 * top_features), fet_names, rotation=60)
	plt.show()
	



def classify(x, y):
	""" Use multi-class linear SVC for classification """

	boundary = int(0.8*len(x))
	x_train, y_train, x_test, y_test = split_train_test(x,y,boundary)
	#print(Counter(y_train))
	# classify once
	vec = TfidfVectorizer(preprocessor=identity, ngram_range=(3,6), analyzer='char_wb')
	#print(vec.get_feature_names())
	#vec = TfidfVectorizer(preprocessor=identity, ngram_range=(1,3))

	feats = generate_feats()

	feature_processing = Pipeline([('feats', FeatureUnion(feats))])
	feature_processing.fit_transform(x_train)

	cls = LinearSVC(C=200)
	pipeline = Pipeline([
		('feats', FeatureUnion(feats)),
		('cls', cls)
	])

	
	classifier = pipeline




	#c = [200,300,500,750,1000]
	"""classifier = Pipeline([
		('features', FeatureUnion(
			feats
		)),
		('clf', LinearSVC(C=200))])"""
	#for i in range(len(feats)):
			#for i in c:



	for j in range(10):
		#pipeline.fit(X_train, y_train)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print(accuracy_score(y_test,y_pred))


	#conf_matrix = confusion_matrix(y_test, y_pred)
	#print("\n\n",classification_report(y_test, y_pred))
	#print_top10(x_train, y_train)
	#most_informative_features(vec,clsf)
	# k-fold cross validation to get an average
	#scores = cross_val_score(classifier, x_train, y_train, cv=10, scoring="accuracy") 
	#print("\nAverage macro f1 score with K-Vold (k=10): {0}\n ".format(round(sum(scores) / len(scores),3)))
	#print("\nAverage macro f1 score with K-Vold (k=10) with feature {0}: {1}\n ".format(feats[i][0],round(sum(scores) / len(scores),3)))
	#print(conf_matrix)
	#try_vis(conf_matrix, y_test, y_pred)


def oneVSone(x,y,balancing):
	boundary = int(0.8*len(x))
	x_train, y_train, x_test_original, y_test = split_train_test(x,y,boundary)

	artist_combinations = list(combinations(set(y),2)) # [(artist1,artist2),(artist1,artist3).. etc.]

	artist_dict = defaultdict(list)
	artist_dict_predict = {}

	for song, artist in zip(x_train, y_train): 
		artist_dict[artist].append(song) # artist_dict to know which songs must be trained on per artist
		artist_dict_predict[artist] = []	# artist_dict_predict to keep track which tracks are predicted to belong to which artist

	i = 0
	for artist1, artist2 in artist_combinations:
		if i == 0: 
			# in the first iterartion all songs must be classified
			x_test = x_test_original
		else:
			# only predict song which are assigned to an artist
			x_test = artist_dict_predict[artist1] + artist_dict_predict[artist2] 
		i += 1
		if balancing.lower() == 'false': 
			# use all songs of the two artists
			x_train = artist_dict[artist1] + artist_dict[artist2]
			y_train = [artist1] * len(artist_dict[artist1]) + [artist2] * len(artist_dict[artist2])
		else: 
			# makes the two artists have the same amount of songs
			max_songs = min([len(artist_dict[artist1]), len(artist_dict[artist2])]) 
			x_train = artist_dict[artist1][:max_songs] + artist_dict[artist2][:max_songs]
			y_train = [artist1] * max_songs + [artist2] * max_songs

		# classify
		vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(3,6), analyzer='char_wb')
		clsf = svm.LinearSVC()
		classifier = Pipeline([('vec', vec), ('cls', clsf)])
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		

		for song,artist in zip(x_test,y_pred):
			# assign song to predicted artist
			if song not in artist_dict_predict[artist]:
				artist_dict_predict[artist].append(song) 
			# remove song if assigned to any other artist
			for artist_in_dict, songs in artist_dict_predict.items():
				if artist != artist_in_dict:
					if song in songs:
						temp_songs = songs
						temp_songs.remove(song)
						artist_dict_predict[artist_in_dict] = temp_songs

	# generate the y_pred in the order of x_test	
	y_pred = []
	for song in x_test_original:
		for artist,songs in artist_dict_predict.items():
			if song in songs:
				y_pred.append(artist)


	return y_test, y_pred


def ten_times_1v1(original_x,original_y,balancing):
	""" Classifies 10 times with the 1v1 method. Each time with a newly shuffled X and Y """

	accuracy_scores = 0
	for i in range(10):
		x_and_y = [(x,y) for x,y in zip(original_x,original_y)] # combine x and y to keep the correct labels linked to the lyrics
		x, y = shuffe_x_and_y(x_and_y)
		y_test, y_pred = oneVSone(x,y,balancing)
		accuracy_scores += accuracy_score(y_test,y_pred)

	print("\n\n",classification_report(y_test, y_pred)) # print out the last classification report
	print("\nAverage accuracy score after 10 iterations with the 1v1 method: {0}\n".format(round(accuracy_scores/10,3)))
	# print confusion matrix of the last iteration
	conf_matrix = confusion_matrix(y_test, y_pred)
	try_vis(conf_matrix,y_test, y_pred)

def parse():
	""" Set up the argparse """

	parser = argparse.ArgumentParser(\
		description='Author attribution on lyrics. The first argument in optional arguments is the default one')
	parser.add_argument('-m', '--method', type=str, metavar='', \
		help="choose the classification method:'multi' or '1v1'",default='multi')
	parser.add_argument('-a','--artists', type=str, metavar='', \
		help="choose which data set to use:'diverse' or'afro_male'", default='diverse')
	parser.add_argument('-b', '--balancing', type=str, metavar='', \
		help="choose to balance the amount of songs in 1v1 classification: 'true' or 'false'", default ="true")
	return parser.parse_args()

def convert_to_path(artists):
	""" Converts argparse choice to path for right data set """

	if artists == "afro_male":
		return "../lyrics/afro_males/"
	else:
		return "../lyrics/diverse/"


def print_top10(x_train,y_train):
	"""Prints features with the highest coefficient values, per class"""
	vec = TfidfVectorizer(ngram_range=(3, 6), analyzer='char_wb')
	vec = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
	x_train = vec.fit_transform(x_train)
	
	classifier = LinearSVC(C=200)
	classifier.fit(x_train, y_train)
	class_labels = classifier.classes_
	feature_names = vec.get_feature_names()
	for i, class_label in enumerate(class_labels):
		top10 = np.argsort(classifier.coef_[i])[-10:]
		features = [feature_names[j] for j in top10]
		print("{0}:{1}".format(class_label, features))
		#print("{0}: {1}".format(class_label,
			  #" ".join(feature_names[j] for j in top10)))

def main():
	""" Call as follows: python3 multi_class.py *path to lyrics files*. Type python3 multiclass.py -h for help """
	args = parse()
	x,y = balance_afro_male_data_to_diverse()
	path = convert_to_path(args.artists)
	#x,y = import_data(path)
	print(set(y))
	if args.method =="1v1":
		ten_times_1v1(x,y, args.balancing)
	else:
		#x,y = convert_to_gender_classification(x,y)
		#most_informative_features(x,y)
		classify(x,y)

if __name__ == "__main__":
	main()