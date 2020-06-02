#!/usr/bin/env python3
# preprocess_lyrics.py
# Author: Tomer Gabay
# January - May 2020

from collections import Counter, defaultdict
from itertools import combinations
from spacy import displacy

import pandas as pd
import numpy as np
import regex as re

import en_core_web_lg
import phonetics
import warnings
import argparse
import inflect
import pyphen
import pickle
import string
import random
import spacy
import nltk
import math
import sys
import csv
import os

warnings.filterwarnings('ignore')


def get_nicknames():
	
	""" Returns a dictionairy of nicknames per artist, based on Wikipedia """

	return {"JAY-Z":["Jay-Z","Jay","Hova","HOV","hov","Hov","Jigga","Shawn Carter","Shawn","Carter"],
	"Eminem": ["Eminem","Marshall Mathers","Marshall","Mathers","Slim Shady","Slim","Shady"],
	"Future": ["Future","Nayvadius Wilburn","Neyvadius","Wiburn","Meathead","Caeser Lee","Ceaser","Lee"],
	"Ice Cube": ["Ice Cube","Ice","Cube","O'Shea Jackson","O'Shea","Jackson"],
	"Lil’ Kim": ["Lil’ Kim","Lil’","Kim","Kimberley Jones","Kimberley","Jones","Queen Bee","Queen","Bee", "Lil'","Lil' Kim","own_nameme", "own_name own_name"],
	"Machine Gun Kelly": ["Machine Gun Kelly","Machine Gun","Gun Kelly","Kelly","Kells","Richard Baker","Richard","Baker","MGK"],
	"Nas": ["Nasty Nas","Nasty","Nas","Escobar", "Jones"],
	"Nicki Minaj": ["Nicki Minaj","Nicki","Minaj","Onika Maraj","Onika","Maraj"],
	"50 Cent": ["50 Cent","Fifty Cent","fifty cent","fifty","fiftycent","50","Cent","Ferrari F-50","Ferrari","F-50","Curtis Jackson","Curtis","Jackson"],
	"2Pac": ["2Pac","twopac","Tupac Shakur","Tupac","Shakur","Makaveli","MC New York", "Pac"],
	"Lil Wayne": ["Lil Wayne","Wayne","Tunechi","Weezy F. Baby", "Weezy","President Carter","Dwayne Carter","Dwayne","Carter"],
	"Snoop Dogg": ["Snoop Dogg","Snoop","Doggy","Dogg","DJ Snoopadelic","Snoopadelic","Niggarachi","Snoopzilla","Nemo Hoes","Nemo"],
	"Damian Marley": ["Damian Marley","Damian","Robert","Nesta","Jr. Gong","Jr Gong","Junior Gong","Gong","Junior","Jr."],
	"Kanye West": ["Kanye West","Kanye","West","Yeezy","\bYe\b", "Omari"],
	"Cardi B": ['Cardi B','Cardi','\bB\b','Belcalis','Marlenis','Alamanzar'],
	"MC Lyte": ['MC Lyte','Lyte','Lana','Michelle','Moorer'],
	"Missy Elliott": ['Missy Elliot','Missy','Elliot','Misdemeanor','Melissa','Arnette'],
	"Iggy Azalea": ['Iggy Azalea','Iggy','Azalea','Amethyst','Amelia','Kelly'],
	"Queen Latifah": ['Queen Latifah','Queen','Latifah','Dana','Elaine','Owens']
	}


def get_word_count(lyrics):


	""" Returns the total word count of a string """

	lyrics = re.sub("['’]"," ",lyrics) # to convert e.g. I'm into I m
	lyrics = lyrics.translate(str.maketrans('','',string.punctuation))
	return len(lyrics.split())
	
def get_sentence_count(lyrics):

	""" Returns the sentence count of a string """

	return len(lyrics.split('\n'))

def get_avg_word_length(lyrics):

	""" Returns the average word length of a string """

	lyrics = lyrics.translate(str.maketrans('','',string.punctuation))
	return round(sum([len(word) for word in lyrics.split()]) / len(lyrics.split()),2)

def get_unique_word_ratio(lyrics):

	""" Returns the unique word ratio of a string """

	lyrics = re.sub("['’]"," ",lyrics)
	lyrics =lyrics.translate(str.maketrans('','',string.punctuation))
	return round(len(set(lyrics.split())) / len(lyrics.split()),2)


def get_repeated_sentence_ratios(lyrics):

	""" Returns two differently calculated repeated sentence ratios """

	repeated_sentence_count_ratios = [] # sum of sentences that are repeated / amount of sentences
	repeated_sentence_ratios = [] # sum of different sentences that are repeated / amount of different sentences
	sentence_counter = Counter(lyrics.split('\n'))
	total_sentences = len(lyrics.split('\n'))
	repeated_sentences_count = sum([instances for sentence,instances in sentence_counter.items()])
	repeated_sentences = sum([1 for sentence,instances in sentence_counter.items()])
	return round(repeated_sentences_count/total_sentences,2), round(repeated_sentences/len(sentence_counter),2)
							

def create_phonetic_representation(lyrics):

	""" Returns the phonetic representation of string """

	to_phonetics = nltk.corpus.cmudict.dict()
	phonetics_repr = ''
	lyrics = lyrics.lower()
	lyrics = re.sub("' "," ",lyrics) # to convert words as runnin' to runnin
	lyrics = re.sub("\-"," ",lyrics) # convert words a four-door to four door
	for word in lyrics.lower().split():
		try:
			phonetics_repr += "".join(to_phonetics[word][0]) + ' '
		except:
			pass # pass if the word is not in the dictionairy
	return phonetics_repr.rstrip()
	
	
def create_soundex_representation(lyrics):

	""" Returns the Soundex representation of a string """

	soundex_repr = ''
	for word in lyrics.split():
		try:
			soundex_repr += phonetics.soundex(word) + ' '
		except:
			word = re.sub("'","",word)
			words = re.sub("\-", " ",word) # convert words a four-door to four door
			for word in words.split():
				try:
					soundex_repr += phonetics.soundex(word) + ' '
				except:
					pass
	return soundex_repr.rstrip()

def create_metaphone_representation(lyrics):

	""" Returns the Metaphone representation of a string """

	metaphone_repr = ''
	for word in lyrics.split():
		try:
			metaphone_repr += phonetics.metaphone(word) + ' '
		except:
			print(word)
	return metaphone_repr.rstrip()


def create_length_representation(lyrics): # based on 'Bleaching text: Abstract features for cross-lingual gender prediction', van der Goot et al. 2018
	
	""" Returns a length representation, in which words are represented by their length, e.g.: Hello PC --> 05 02 """
	
	length_repr = ''
	for sentence in lyrics.split('\n'):
		sentence_repr = ''
		for word in sentence.split():
			sentence_repr += '0' + str(len(word)) + ' '
		length_repr += sentence_repr.rstrip() + '\n' # add newline to preserve line structure
	
	return length_repr.rstrip()

def create_punctC_representation(lyrics): # based on 'Bleaching text: Abstract features for cross-lingual gender prediction', van der Goot et al. 2018
	
	""" Returns a representation in which punctuation is preserved """
	
	punctC_repr = ""
	for sentence in lyrics.split('\n'):
		sentence_repr = ''
		for word in sentence.split():
			punctC = ""
			for char in word:
				if char not in string.punctuation:
					punctC += 'W'
				else:
					punctC += char
			punctC = re.sub("W+", "W", punctC) + ' '
			sentence_repr += punctC
		punctC_repr += sentence_repr.rstrip() + '\n'
		
	return punctC_repr.rstrip()

def create_shape_representation(lyrics): # based on 'Bleaching text: Abstract features for cross-lingual gender prediction', van der Goot et al. 2018
	
	""" Returns a representation which is based on capitality of letters, and digits"""
	
	shape_repr = ''
	for sentence in lyrics.split('\n'):
		sentence_repr = ''
		for word in sentence.split():
			shape = ''
			for char in word:
				if char.isupper():
					shape += 'U'
				elif char.islower():
					shape += 'L'
				elif char.isdigit():
					shape += 'D'
				else:
					shape += 'X'
			for letter in 'ULDX':
				shape = diminish_duplicate_letters(shape,letter)
			sentence_repr += shape + ' '
		shape_repr += sentence_repr.rstrip() + '\n'
	return shape_repr.rstrip()
				
def diminish_duplicate_letters(chars,char): 

	""" Converts a 3 or more idental consecutive letters to 2 """

	return re.sub(char +"{3,}",char+char,chars)

def create_vowel_representation(lyrics): # based on 'Bleaching text: Abstract features for cross-lingual gender prediction', van der Goot et al. 2018
	
	""" Returns a representation based on vowels """
	
	vowel_representations = ''
	for sentence in lyrics.split('\n'):
		sentence_repr = ''
		for word in sentence.split():
			vowel_repr = ''
			for char in word:
				if char.lower() in 'aeiou':
					vowel_repr += 'V'
				elif char.lower() in 'bcdfghjklmnpqrstvwxyz':
					vowel_repr += 'C'
				else:
					vowel_repr += 'O'
			sentence_repr += vowel_repr + ' '
		vowel_representations += sentence_repr.rstrip() + '\n'
	return vowel_representations.rstrip()


def create_syllable_representation(lyrics):

	""" Returns a syllable representation of a string """

	lyrics.translate(str.maketrans('', '', string.punctuation)) # source: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
	to_syllables = pyphen.Pyphen(lang='en')
	syllable_representation = ''
	for sentence in lyrics.split('\n'):
		words = sentence.split()
		for word in words:
			syllables = to_syllables.inserted(word)
			syllable_representation += re.sub("\-", " ", syllables) + ' '
		syllable_representation += '\n'
	return syllable_representation.rstrip()

def create_NER_representation(lyrics, nlp):
	
	""" Returns a representation based on Named Entity Recognition """
	
	NER_repr = ''
	for sentence in lyrics.split('\n'):
		NER = [(X.text,X.label_) for X in nlp(sentence).ents] # create tuples of the word and its eventual NER tag
		for word in sentence.split():
			added_NER = False
			for tupl in NER:
				if word == tupl[0]:
					NER_repr += tupl[1]
					added_NER = True
			if added_NER == False:
				NER_repr += word
			NER_repr += ' '
		NER_repr.rstrip()
		NER_repr += '\n'
			
	return NER_repr.rstrip()

def create_POS_representation(lyrics):
	
	""" Creates a representation based on POS tagging """
	
	tokens = nltk.word_tokenize(lyrics)
	pos_tags = [output[1] for output in nltk.pos_tag(tokens)]
	return ' '.join(pos_tags)

def number_to_word(number):  

	""" Converts a number to its word representation, e.g. 50 to fifty"""

	return inflect.engine().number_to_words(number.group(1))


def get_artist_list(data_set):

	""" Returns a list with the artist names of a dictionairy """

	return [dictio['artist'] for dictio in data_set]


def preprocess(data,nlp):
	
	""" Returns a preprocessed x """
	
	inflect_engine = inflect.engine()
	nicknames = get_nicknames()
	print("Total instances to preprocess: {}".format(len(data)))
	i = 0 # to track where the program is
	

	new_data = []

	for dictio in data:
		lyrics = dictio['lyrics']
		artist = dictio['artist']
		lyrics = re.sub("\[.*\]", "", lyrics) # removes info like [Intro: Eminem]
		lyrics = re.sub("\*.*?\*", "", lyrics) # text between *..* usually announces something
		lyrics = re.sub("[wW]\/","", lyrics) # marker for w/ --> with
		lyrics = re.sub("[Cc]horus","", lyrics)
		lyrics = re.sub("[Vv]erse","",lyrics)
		lyrics = re.sub("[xX][1-9]","",lyrics) # marker for repeated structures
		lyrics = re.sub("\n+","\n", lyrics) # replaces multiple newlines by a single newline
		lyrics = re.sub("\{\}\[\]\*\&", "", lyrics) # remove irrelevant punctuation
		for nickname in nicknames[dictio['artist']]: # replaces artists' nicknames with 'own_name' placeholder
			lyrics = re.sub(nickname,"own_name",lyrics)
			lyrics = re.sub(nickname.lower(),"own_name",lyrics)
			lyrics = re.sub(nickname.upper(),"own_name",lyrics)
		
		lyrics = re.sub(" .*?own_name.* "," own_name ",lyrics) # replaces e.g. own_namey with own_name
	   
		dictio['shape_repr'] = create_shape_representation(lyrics) # create shape representation before converting digits to letters
		lyrics = re.sub("\.([1-9])",r'\1',lyrics) # convert .9 to 9
		dictio['pos_repr'] = create_POS_representation(lyrics) # create POS representation before converting digits to letters
		lyrics = re.sub(" 911", " 9 1 1",lyrics)
		lyrics = re.sub("19([0-9]{2})",r'19 \1',lyrics)
		lyrics = re.sub("([0-9]+)",number_to_word,lyrics) # convert numbers to words e.g. 50 to fifty
		dictio['word_count'] = get_word_count(lyrics)
		dictio['sentence_count'] = get_sentence_count(lyrics)
		dictio['avg_word_length'] = get_avg_word_length(lyrics)
		dictio['unique_word_ratio'] = get_unique_word_ratio(lyrics)
		dictio['repeated_sentence_count_ratio'], dictio['repeated_sentence_ratio'] = get_repeated_sentence_ratios(lyrics)
		
		lyrics = re.sub("([ \n])[\'\"\*\’\:\;\(\)]",r'\1',lyrics) # removes specific punctuation after a space
		lyrics = re.sub("[\'\"\*\’\:\;\(\)]([ \n])",r'\1',lyrics) # removes specific punctuation before a space

		dictio['lyrics'] = lyrics

		# create different representations
		dictio['syllab_repr'] = create_syllable_representation(lyrics)
		dictio['length_repr'] = create_length_representation(lyrics)
		dictio['punctC_repr'] = create_punctC_representation(lyrics)
		dictio['vowel_repr'] = create_vowel_representation(lyrics)
		dictio['syllab_repr'] = create_syllable_representation(lyrics)
		dictio['metaphone_repr'] = create_metaphone_representation(lyrics)
		dictio['soundex_repr'] = create_soundex_representation(lyrics)
		dictio['phonetic_repr'] = create_phonetic_representation(lyrics)

		lyrics = re.sub("own_name","John", lyrics) # convert own_name to John for better NER_tagging
		dictio['ner_repr'] = create_NER_representation(lyrics,nlp)
		
		new_data.append(dictio)
		
		# to track where to program is while running
		i += 1 
		if i % 100 == 0:
			print(i,end=' ')

	return new_data


def convert_to_verse_classification_duo_artist(data):
	
	""" Convert to verse classification in a duo artist songs in which artist is set as e.g.: Jay-Z & Kanye West"""
	
	nicknames = get_nicknames()
	artist1, artist2 = data[0]['artist'].split('&')[0].strip(), data[0]['artist'].split('&')[1].strip() # get the two artist names
	new_data = []
	for dictio in data:
		del dictio['featuring']
		lyrics = dictio['lyrics']
		lyrics = re.sub("\n","___",lyrics)
		verses = re.findall("\[.+?\].+?\[",lyrics,overlapped=True)
		verses = [re.sub("___","\n",verse) for verse in verses]
		verses = [re.sub("\n+\[","",verse) for verse in verses]
		for verse in verses:
			y_verse = "OTHER ARTIST" # in case more artists particiate than artist1 and artist2
			header = re.findall("\[.+?\]",verse)[0].lower() # header of a verse, as in [..]
			verse = re.sub("\[.+?\]","",verse)
			if header != []:
				header = header.split(':') # usually headers are like [verse1: artist]
				if len(header) > 1:
					header = header[1].strip()[:-1]
				elif type(header) == list: # this means the header didn't have a :
					header = header[0].split('-') # sometimes headers are like [verse1 - artist]
					if len(header) > 1:
						header = header[1].strip()[:-1]
				for nickname in nicknames[artist1]:
					if header == nickname.lower():
						y_verse = artist1
				for name in nicknames[artist2]: # set verse to artist2 of its not set to artist 1 or combined verse yet
					if name.lower() == header and y_verse != artist1 and y_verse != 'combined verse':
						y_verse = artist2
				if y_verse == artist1 or y_verse == artist2:
					if len(verse.split()) >= 20:
						new_dictio = dictio.copy()
						new_dictio['artist'] = y_verse
						new_dictio['lyrics'] = verse
						new_data.append(new_dictio)
			
	return new_data


def convert_to_verse_classification(data):
	
	""" Converts instances of songs to instances of verses """
	
	nicknames = get_nicknames()
	new_data = []

	for dictio in data:
		artist = dictio['artist']
		lyrics = dictio['lyrics']
		lyrics = re.sub("\n","___",lyrics) # replace by ___ to preserse the location of the newline
		verses = re.findall("\[.+?\].+?\[",lyrics,overlapped=True) # [...] indicates the start of a new verse
		verses = [re.sub("___","\n",verse) for verse in verses] # reinsert the newlines
		verses = [re.sub("\n+\[","",verse) for verse in verses] # remove a remaining [
		for verse in verses:
			if isinstance(dictio['featuring'],float): # if the entire song is by the same artist, simply add each verse to the data
				verse = re.sub("\[.+?\]","",verse)
				if len(verse.split()) > 20:
					new_dictio = dictio.copy()
					new_dictio['lyrics'] = verse.strip()
					if new_dictio not in new_data:
						new_data.append(new_dictio)
						all_verses.append(verse.strip())
			else: # if the song in by multiple artists, check the artist of each verse
				header = re.findall("\[.+?\]",verse) # header of a verse, as in [..]
				if header != []:
					header = header[0].lower()
					header = header.split(':')
					if len(header) > 1:
						header = header[1].strip()[:-1]
					for nickname in nicknames[dictio['artist']]:
						if header == nickname.lower():
							verse = re.sub("\[.+?\]","",verse)
							if len(verse.split()) > 20:
								new_dictio = dictio.copy()
								new_dictio['lyrics'] = verse.strip()
						  
								if new_dictio not in new_data:
									new_data.append(new_dictio)
   
	song_titles = [dictio['song_title'] for dictio in new_data] # add song titles to the data to be able to trace the original song of each verse

	return new_data



def form_x_of_songs_and_verses(data):

	""" Returns a list of dicionaries with songs and verses """

	nicknames = get_nicknames()
	new_data = []
	for dictio in data:
		if isinstance(dictio['featuring'],float): # nan is a float, thus if no featuring artists it's a flaot
			dictio['type'] = 'song'
			new_data.append(dictio)
		else: # if the instance has featuring artists, detect which verses belong to the relevant artist
			lyrics = dictio['lyrics']
			lyrics = re.sub("\n","___",lyrics) # temporarily substitute newline by three underscores for verse identification 
			verses = re.findall("\[.+?\].+?\[",lyrics,overlapped=True) # find the text between:"[..]" and "[", as that is a verse
			verses = [re.sub("___","\n",verse) for verse in verses]
			verses = [re.sub("\n+\[","",verse) for verse in verses]
			combined_verses = []
			for verse in verses:
				header = re.findall("\[.+?\]",verse) # header of a verse, as in [..]
				if header != []:
					header = header[0].lower() # lower header for generalisability 
					header = header.split(':') # usually a head is [Verse 1: name]
					if len(header) > 1:
						header = header[1].strip()[:-1]
					for nickname in nicknames[dictio['artist']]:
						if header == nickname.lower(): # lower nickname for generalisability
							verse = re.sub("\[.+?\]","",verse)
							combined_verses.append(verse)
							break
			combined_verses = "\n".join(combined_verses)
			if len(combined_verses.split()) > 20:
				dictio['lyrics'] = combined_verses
				dictio['type'] = 'verses'
				new_data.append(dictio)
	return new_data


def import_duo_artist_file(path):

	""" Returns a list of song dictionaires of a specific path-defined csv file"""

	for filename in os.listdir(path):
		if filename[-4:] == ".csv" and 'dev' not in filename and 'train' not in filename and 'test' not in filename and '&' in filename:
			df = pd.read_csv(path+filename)
	data = []
	for i,row in df.iterrows():
		data.append({"song_title":row["song_title"],"artist":row['artist'],"lyrics":row['lyrics'],"featuring":row['featuring']})
	return data

def import_artist_files(path):

	""" Imports the data per artist file and returns a list with dictionaires of songs """

	songs_per_artist = []
	for filename in os.listdir(path):
		if filename[-4:] == ".csv":
			if 'dev' not in filename and 'train' not in filename and 'test' not in filename and '&' not in filename:
				songs_per_artist.append(pd.read_csv(path+filename))
	df = pd.concat(songs_per_artist, ignore_index = True) # concatenate all the panda dataframes
	data = []
	for i,row in df.iterrows():
	   data.append({"song_title":row["song_title"],"artist":row['artist'],"lyrics":row['lyrics'],"featuring":row['featuring']})

	return data

def write_to_csv(data, data_type, dataset):

	""" Write the list of dictionaires to a csv file """

	keys = list(data[0].keys())
	with open("../datasets/"+ dataset + "_" + data_type + ".csv", 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(data)


def split_train_dev_test(data):

	""" Returns a dictionairy with train dev and test data """

	songs = []
	verses = []
	for dictio in data:
		if isinstance(dictio['featuring'],float): # nan is a float, thus if no featuring artists it's a float
			songs.append(dictio)
		else:
			verses.append(dictio)
	random.seed(50)
	random.shuffle(songs)
	train = songs[:int(0.8*len(songs))] # training data consists of 80%
	dev = songs[int(0.8*len(songs)):int(0.9*len(songs))] # development data consists of 10%
	test = songs[int(0.9*len(songs)):] # test data consists of 10%
	#if add_verses == True:
	train = train + verses # add verses to the training data
	random.shuffle(train)
	return {"train":train,"dev":dev,"test":test}


def load_one_gender_data(gender,path):

	""" Returns the train, dev and test data of one gender """

	data = import_artist_files(path+gender+"/")
	for dictio in data:
		dictio['gender'] = gender
	datas = split_train_dev_test(data)
	return datas


def balance_data_set(data_set_l, data_set_s,mc_artist_l,mc_artist_s):    
		
	""" Returns the largest dataset, completely balanced to the smaller, based on the most common instances of each data set """

	y_data_set_s = Counter(get_artist_list(data_set_s))
	new_data_set_l = []
	for l,s in zip(mc_artist_l,mc_artist_s):
		artist_l = l[0] # name of the artist of the larger data set
		songs_s = y_data_set_s[s[0]] # amount of songs in the smaller data set
		new_data_set_l += [dictio for dictio in data_set_l if dictio['artist'] == artist_l][:songs_s] # assign as many instances of the larger data set as the smaller has, per artist
	return new_data_set_l



def return_songs_and_verses_seperately(datas):

	""" Returns train, dev, test with only songs and a list with only verse instances """

	songs = {}
	for data_type in datas.keys():
		songs[data_type] = [dictio for dictio in datas[data_type] if isinstance(dictio['featuring'],float)]
	verses = [dictio for dictio in datas["train"] if not isinstance(dictio['featuring'],float)]
	return songs,verses

def create_datasets_for_GBDS(path):

	""" Returns a completely on artist balanced data set with male and female labels """

	# load in the train dev and test per gender
	datas_female = load_one_gender_data("female",path)
	datas_male = load_one_gender_data("male",path)
	
	# determine the amount of songs and verses per gender
	datas_female = {data_type:form_x_of_songs_and_verses(data) for data_type,data in datas_female.items()}
	datas_male = {data_type:form_x_of_songs_and_verses(data) for data_type,data in datas_male.items()}
	
	# split the songs and verses per gender
	songs_female, verses_female = return_songs_and_verses_seperately(datas_female)
	songs_male, verses_male = return_songs_and_verses_seperately(datas_male)
	mc_songs_female = Counter(get_artist_list(songs_female['train'])).most_common() # the balancing is based on the amount of training songs per artist
	mc_songs_male = Counter(get_artist_list(songs_male['train'])).most_common()

	datas = {}
	for (key_female,songs_female), (key_male,songs_male) in zip(datas_female.items(),datas_male.items()): # the keys are either train dev or test
		male_songs_balanced_to_female = balance_data_set(songs_male,songs_female,mc_songs_male,mc_songs_female)
		datas[key_female] = male_songs_balanced_to_female + songs_female # add the male and female songs to the data set
	
	male_verses_balanced_to_female = balance_data_set(verses_male,verses_female,mc_songs_male,mc_songs_female) # balance the amount of verse of males to female
	datas['train'] += male_verses_balanced_to_female + verses_female # add the verses to the training set
	return datas

def check_if_should_run(args):

	""" check if the program should run based on the previous outcome and the overwrite argument """

	if args.overwrite == False:
		try:
			x_train = pd.read_csv("../datasets/"+args.dataset+"_train.csv")
			print("Outcome already stored in: .../datasets/"+args.dataset+"_'train/dev/test'.csv")
			print("To rerun and overwrite, use -ow True. This could take a few hours maxiumum.")
			return False
		except:
			pass
	print("Preprocessing lyrics. This could take a few hours maxiumum.")
	return True

def parse():

	""" Set up the argparse """

	parser = argparse.ArgumentParser(\
		description='Preprocessing on lyrics. Select data set with -ds argument')
	parser.add_argument('-ds','--dataset', type=str, metavar='', \
		help="choose which data set to use: 'DADS', 'AAMDS','GBDS', 'CADSa' or 'CADSb'", required=True)
	parser.add_argument('-ow', '--overwrite', type=str2bool,metavar='', \
		help="overwrite the previoulsy determined outcome, choose true or false. Default is false", default=False)
	return parser.parse_args()

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


def main():
	args = parse()
	run = check_if_should_run(args)

	if run == True:
		nlp = en_core_web_lg.load()
		

		path = "../lyrics/"+args.dataset+"/"

		if args.dataset != "GBDS":
			data = import_artist_files(path)
			datas = split_train_dev_test(data)
			datas = {data_type:form_x_of_songs_and_verses(data) for data_type,data in datas.items()}
		else:
			datas = create_datasets_for_GBDS(path)

		if "CADS" in args.dataset:
			datas = {'train':datas['train']+datas['dev']+datas['test']}
			
			data = import_duo_artist_file(path)
			data = convert_to_verse_classification_duo_artist(data)
			data = preprocess(data,nlp)
			write_to_csv(data,"test",args.dataset)
		datas = {data_type:preprocess(data,nlp) for data_type,data in datas.items()}
		for data_type, data in datas.items():
			write_to_csv(data,data_type,args.dataset)
		print("Succesfully preprocessed all lyrics to .../datasets/"+args.dataset+"_'train/dev/test'.csv")

if __name__ == "__main__":
	main()