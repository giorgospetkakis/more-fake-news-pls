## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import re
import spacy
import numpy as np

def __clean_tweets__(tweets, pattern=None, clean_USER=False):
	pattern = r"HASHTAG|URL|RT|#|https:\/\/t\.\\[a-z\d]+|https.*$|\&*amp"
	if clean_USER:
		pattern += r"|USER"
	return [re.sub(pattern, "", tweet) for tweet in tweets]


def extract_semantic_similarity(authors, model=None):
	'''
	Generates the semantic similarity for the given authors and returns the dict.
	
	Parameters:
		authors(dict):
			The given dictionary of authors to be processed.
		
		model(spaCy model):
			The spaCy model to load for feature extraction.

	Returns:
		Dict of authors with Semantic Similarity features added
	'''
	if model == None:
		print("Loading spacy data...")
		model = spacy.load("en_core_web_md")

	print("Extracting semantic similarity. This may take some time...")
	for author in authors.keys():

		# Prepare our array to hold the similarity values for each author
		# Initialize values as -1 so that we can easily separate out 'empty' values later
		a = np.ones((100,100))
		a *= a*-1
		i = 0

		# Pipe to go through the documents faster
		for tweet in model.pipe(authors[author].tweets):
		    j = i+1
		    if i == 99:
		        break
		    for compare in model.pipe(authors[author].tweets[i+1:]):
		        if j == 100:
		            break
		        a[i,j] = tweet.similarity(compare)
		        j += 1
		    i += 1
		authors[author].similarities = a
		authors[author].max_similar = a[ a != 1].max()
		authors[author].min_similar = a[ a != -1].min()
		authors[author].mean_similar = a[ a != -1].mean()
		authors[author].number_identical = len(np.where(a ==1.0)[0])
		print('|',end=' ')

	print("")
	print("Done")
	return(authors)

def extract_pos_tags(authors, model=None, ignore=["HASHTAG", "URL"]):
	'''
	Generates the Part-Of-Speech tags for the given authors and returns the dict.
	
	Parameters:
		authors(dict):
			The given dictionary of authors to be processed.
		
		model(spaCy model):
			The spaCy model to load for feature extraction.
		
		ignore(list):
			The list of terms to ignore.
	Returns:
		Dict of authors with Part-Of-Speech tags added
	'''

	# Load spacy model
	if model == None:
		print('Loading spacy data...')
		model = spacy.load("en_core_web_md")
	
	print('Processing Part-Of-Speech Tags...')
	# Go through each author
	for author in authors.keys():
		
		# Clean up tweets
		tweets_cleaned = __clean_tweets__(authors[author].tweets)

		# Pipe to go through the documents faster
		for tweet in model.pipe(tweets_cleaned, disable=['parser']):
			#Collect and save tags
			tags = []
			tags.append(token.pos_ for token in tweet if token.text not in ignore)
			authors[author].POS_tags.append(tags)
	print('Done')
	return(authors)

def extract_named_entities(authors, model=None):
	'''
	Generates the named entities for the given authors and returns the dict.
	
	Parameters:
		authors(dict):
			The given dictionary of authors to be processed.
		
		model(spaCy model):
			The spaCy model to load for feature extraction.
		
		ignore(list):
			The list of terms to ignore.
	Returns:
		Dict of authors with named entities added
	'''

	# Load spacy model
	if model == None:
		print('Loading spacy data...')
		model = spacy.load("en_core_web_md")
	
	print('Processing Named Entities...')
	# Go through each author
	for author in authors.keys():
		
		# Clean up tweets
		tweets_cleaned = __clean_tweets__(authors[author].tweets, clean_USER=True)

		# Pipe to go through the documents faster
		for tweet in model.pipe(tweets_cleaned, disable=['parser']):

			# Named Entity Recognition
			authors[author].ents.append([[str(ent.text), str(ent.label_)] for ent in list(tweet.ents)])
	
	print('Done')
	return(authors)

def extract_clean_tweets(authors, model=None, ignore=["HASHTAG", "URL"]):
	'''
	Generates the cleaned, lemmatized tweets for the given authors and returns the dict.
	
	Parameters:
		authors(dict):
			The given dictionary of authors to be processed.
		
		model(spaCy model):
			The spaCy model to load for feature extraction.
		
		ignore(list):
			The list of terms to ignore.
	Returns:
		Dict of authors with clean tweets added.
	'''

	# Load spacy model
	if model == None:
		print('Loading spacy data...')
		model = spacy.load("en_core_web_md")
	
	print('Cleaning tweets...')
	# Go through each author
	for author in authors.keys():
		
		# Clean up tweets
		tweets_cleaned = __clean_tweets__(authors[author].tweets)

		# Pipe to go through the documents faster
		for tweet in model.pipe(tweets_cleaned, disable=['parser']):

			# Collect only the lemma form of the words. Only words with only alpha characters kept.
			lemmas = []
			lemmas.append([token.lemma_.lower() for token in tweet if (token.is_alpha and token.text not in ignore)])
			authors[author].clean.append(" ".join(lemmas[0]))

	print('Done')
	return(authors)
