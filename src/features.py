## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import re
import data
import emoji
import spacy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from lexical_diversity import lex_div as ld
import preprocessing


def __clean_tweets__(tweets, pattern=None, clean_USER=False):
	pattern = r"HASHTAG|URL|RT|#|https:\/\/t\.\\[a-z\d]+|https.*$|\&*amp"
	if clean_USER:
		pattern += r"|USER"
	clean_tweets = []
	for tweet in tweets:
		clean = re.sub(pattern, '', tweet)
		clean = emoji.get_emoji_regexp().sub(r'', clean)
		clean = clean.strip()
		clean_tweets.append(clean)

	return clean_tweets

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

		#Updated as previous calculation was incorrect.
		identicals = []
		for i in range(len(authors[author].tweets)):
			if np.where(authors[author].similarities[i] == 1.0)[0].shape[0] > 0:
				identicals.append(i)
				identicals.extend(list(np.where(authors[author].similarities[i] == 1.0)[0]))
		authors[author].number_identical = len(set(identicals))/len(authors[author].tweets)
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

def extract_mcts_ner(authors, n_models=50, k=3, threshold=1.0, _max_iter=5000, _n_init=5, data_split=0.6667):
    '''
    Extract the most common terms used by fake-news spreaders for the given authors.
    Trains an n_models number of models and takes the intersection of the most common terms used by fake news spreaders.
    Uses k-means and tf-idf

    Parameters:
        authors(Author dict):
            The authors to calculate the most common terms for

        n_models(int):
            The number of models to train
            Default is 50.

        k(int):
            The number of clusters for each model
            Default is 3

        _max_iter(int):
            The maximum number of iterations for each model
            Default is 5000

        _n_init(int):
            The number of initializations the k-means model will do. Returns the best one.
            Default is 5

        data_split(float):
            The splitting point between the testing set and training set for the data.
            Default is 60%
    '''

    print("Loading data...")

    # Get truth values
    truths = [auth.truth for auth in list(authors.values())]
    # Get NER values 
    entities = [auth.ents for auth in list(authors.values())]

    # Handle case of empty named entities
    if (entities[0] == []):
        print("Authors contain no named entities. Calculating...")
        authors = extract_named_entities(authors)
        entities = [auth.ents for auth in list(authors.values())]

    # Some cleanup for the vectorization
    for i in range(len(entities)):
        entry = entities[i]
        entities[i] = [e[0] for e in entry if len(e) > 0]

	# Make sure two+ word named entities are handled as single words
    tweets = []
    for i, e in enumerate(entities):
        tweets += [" ".join([re.sub(r" ", "_", x[0]) for x in e])]

    # This will be the list of sets of most common terms used by fake news spreaders
    list_of_sets = []

    # The split of training vs. testing data
    split = int(len(tweets) * data_split)
    tweetsTrain = tweets[:split]
    tweetsTest = tweets[split:]

    print("Creating models...")
    while len(list_of_sets) < n_models:

        # Create model from the vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(tweetsTrain)
        model = KMeans(n_clusters=k, algorithm="full", init='k-means++', max_iter=_max_iter, n_init=_n_init)
        model.fit(X)

        # Order the centroids 
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        # Generate predictions based on the test set
        pred = np.zeros((len(tweetsTest), 2))
        for i, tw in enumerate(tweetsTest):
            X = vectorizer.transform([tw])
            predicted = model.predict(X)
            pred[i] = [int(predicted), truths[i + split]]

        # Fetch predictions results
        res = [(pred[np.where(pred[:, 1]==k)])[:,0] for k in range(2)]
        
        # Calculate the cluster with the maximum purity
        max_purity = 0 ; pure_cluster = -1
        for c in range(k):
            class_a = len(np.where(res[0]==c)[0])
            class_b = len(np.where(res[1]==c)[0])

            class_sum = class_a + class_b
            if class_sum != 0:
                purity = abs(0.5 - class_a / class_sum) * (class_sum/len(tweetsTest)) * 10
            else:
                purity = 0
            if purity > max_purity:
                pure_cluster = c
                max_purity = purity

        # Check if there is a cluster with very high purity
        if max_purity >= threshold:
            list_of_sets += [set()]
            if len(list_of_sets) % (n_models / 4) == 0 and len(list_of_sets) > 0:
                print(f"{len(list_of_sets) / (n_models) * 100}% of requested models trained")
            for ind in order_centroids[pure_cluster,:100]:
                # Add it to the list of sets
                list_of_sets[-1].add(terms[ind])
    
    # Calculate the final set as an intersection of all sets identified
    final_set = list_of_sets[0]
    for s in list_of_sets:
        final_set = final_set.intersection(s)

    for author in authors.keys():
        # Get cleaned values
        cleaned = " ".join(authors[author].clean)
        count = 0

        # Count the number of terms in each author
        for term in list(final_set):
            count += cleaned.count(re.sub("_", " ", term))

        # Save to author
        authors[author].most_common_ner_score = count

    return authors, final_set

def extract_mcts_adj(authors, n_models=50, k=3, threshold=1.0, _max_iter=5000, _n_init=5, data_split=0.6667):
    '''
    Extract the most common terms used by fake-news spreaders for the given authors.
    Trains an n_models number of models and takes the intersection of the most common terms used by fake news spreaders.
    Uses k-means and tf-idf

    Parameters:
        authors(Author dict):
            The authors to calculate the most common terms for

        n_models(int):
            The number of models to train
            Default is 50.

        k(int):
            The number of clusters for each model
            Default is 3

        _max_iter(int):
            The maximum number of iterations for each model
            Default is 5000

        _n_init(int):
            The number of initializations the k-means model will do. Returns the best one.
            Default is 5

        data_split(float):
            The splitting point between the testing set and training set for the data.
            Default is 60%
    '''

    print("Loading data...")

    # Get truth values
    truths = [auth.truth for auth in list(authors.values())]
    # Get ADJ values 
    adjectives = [auth.adjectives for auth in list(authors.values())]

    tweets = []
    for adj in adjectives:
        _str = " "
        for text, count in adj.items():
            _str += " " + (text + " ") * count
        tweets += [_str.strip()]

    # This will be the list of sets of most common adjectives used by fake news spreaders
    list_of_sets = []

    # The split of training vs. testing data
    split = int(len(tweets) * data_split)
    tweetsTrain = tweets[:split]
    tweetsTest = tweets[split:]

    print("Creating models...")
    while len(list_of_sets) < n_models:

        # Create model from the vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(tweetsTrain)
        model = KMeans(n_clusters=k, algorithm="full", init='k-means++', max_iter=_max_iter, n_init=_n_init)
        model.fit(X)

        # Order the centroids 
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        # Generate predictions based on the test set
        pred = np.zeros((len(tweetsTest), 2))
        for i, tw in enumerate(tweetsTest):
            X = vectorizer.transform([tw])
            predicted = model.predict(X)
            pred[i] = [int(predicted), truths[i + split]]

        # Fetch predictions results
        res = [(pred[np.where(pred[:, 1]==k)])[:,0] for k in range(2)]
        
        # Calculate the cluster with the maximum purity
        max_purity = 0 ; pure_cluster = -1
        for c in range(k):
            class_a = len(np.where(res[0]==c)[0])
            class_b = len(np.where(res[1]==c)[0])

            class_sum = class_a + class_b
            if class_sum != 0:
                purity = abs(0.5 - class_a / class_sum) * (class_sum/len(tweetsTest)) * 10
            else:
                purity = 0
            if purity > max_purity:
                pure_cluster = c
                max_purity = purity

        # Check if there is a cluster with very high purity
        if max_purity >= threshold:
            list_of_sets += [set()]
            if len(list_of_sets) % (n_models / 4) == 0 and len(list_of_sets) > 0:
                print(f"{len(list_of_sets) / (n_models) * 100}% of requested models trained")
            for ind in order_centroids[pure_cluster,:100]:
                # Add it to the list of sets
                list_of_sets[-1].add(terms[ind])
    
    # Calculate the final set as an intersection of all sets identified
    final_set = list_of_sets[0]
    for s in list_of_sets:
        final_set = final_set.intersection(s)

    for author in authors.keys():
        # Get cleaned values
        cleaned = " ".join(authors[author].clean)
        count = 0

        # Count the number of terms in each author
        for term in list(final_set):
            count += cleaned.count(re.sub("_", " ", term))

        # Save to author
        authors[author].most_common_adj_score = count

    return authors, final_set

def extract_nonlinguistic_features(authors):
	'''
	Extract the features stored in the nonlinguistic_features dict attribute on the author class.
	Takes dictionary of authors as an input and returns the modified version.
	'''

	for author in authors.keys():

		#Initialize the dictionary with 0's. We will update incrementally
		nonlinguistic_tags = ['url', 'hashtag', 'user', 'emoji', 'exclamation', 'period', 'question', 'comma']
		authors[author].nonlinguistic_features = {
													'mean_words' : 0,
			                                        'retweet_percentage' : 0,
			                                        'allcaps_ratio' : 0,
			                                        'allcaps_inclusion_ratio' : 0,
			                                        'titlecase_ratio'  : 0
			                                        }
		for tag in nonlinguistic_tags:
			authors[author].nonlinguistic_features['{}_mean'.format(tag)] = 0
			authors[author].nonlinguistic_features['{}_max'.format(tag)] = 0

		N = len(authors[author].tweets)

		for tweet in authors[author].tweets: # Looping through all tweets for a given author
			# Use .count() to count the various substrings
			url_count = tweet.count('#URL#')
			hashtag_count = tweet.count('#HASHTAG#')
			user_count = tweet.count('#USER#')
			retweet_count = tweet.count('[RT ')
			word_count = len(list(filter(None, re.split('[ -]', __clean_tweets__([tweet])[0]))))
			nonword_count = url_count + hashtag_count + user_count + retweet_count

			authors[author].nonlinguistic_features['url_mean'] += url_count/N #Divide by the number as you calculate the mean
			if tweet.count('#URL#') > authors[author].nonlinguistic_features['url_max']:
				authors[author].nonlinguistic_features['url_max'] = url_count #Update the max number of URL counts if larger than what is saved
			authors[author].nonlinguistic_features['hashtag_mean'] +=	hashtag_count/N
			if tweet.count('#HASHTAG#') > authors[author].nonlinguistic_features['hashtag_max']:
				authors[author].nonlinguistic_features['hashtag_max'] =	hashtag_count
			authors[author].nonlinguistic_features['user_mean'] += user_count/N
			if tweet.count('#USER#') > authors[author].nonlinguistic_features['user_max']:
				authors[author].nonlinguistic_features['user_max'] = user_count
			authors[author].nonlinguistic_features['exclamation_mean'] += tweet.count('!')/N
			if tweet.count('!') > authors[author].nonlinguistic_features['exclamation_max']:
				authors[author].nonlinguistic_features['exclamation_max'] = tweet.count('!')
			authors[author].nonlinguistic_features['period_mean'] += tweet.count('.')/N
			if tweet.count('.') > authors[author].nonlinguistic_features['period_max']:
				authors[author].nonlinguistic_features['period_max'] = tweet.count('.')
			authors[author].nonlinguistic_features['question_mean'] += tweet.count('?')/N
			if tweet.count('?') > authors[author].nonlinguistic_features['question_max']:
				authors[author].nonlinguistic_features['question_max'] = tweet.count('?')
			authors[author].nonlinguistic_features['comma_mean'] += tweet.count(',')/N
			if tweet.count(',') > authors[author].nonlinguistic_features['comma_max']:
				authors[author].nonlinguistic_features['comma_max'] = tweet.count(',')
			# Not perfect. can use some improvement, but good enough for now. 
			authors[author].nonlinguistic_features['emoji_mean'] += len(re.findall(u'[\U0001f600-\U0001f650]', tweet))/N
			if len(re.findall(u'[\U0001f600-\U0001f650]', tweet)) > authors[author].nonlinguistic_features['emoji_max']:
				authors[author].nonlinguistic_features['emoji_max'] = len(re.findall(u'[\U0001f600-\U0001f650]', tweet))


			# Other linguistic features that do not follow the above pattern
			authors[author].nonlinguistic_features['mean_words'] += (word_count - nonword_count)/N
			authors[author].nonlinguistic_features['retweet_percentage'] += retweet_count/N

			# To calculate capitalization percentage, first count all allcaps substrings, remove all the #USER# #URL# #HASHTAG#
			# Then take the ratio of the adjusted count of the all caps words to the adjusted counts of other words in the tweet
			# Divide by N so that after all of the tweets it will not exceed 100%
			# Only if there are still real words left after removing all the tags (so we don't divide by 0 lol)
			if word_count > nonword_count:
				authors[author].nonlinguistic_features['allcaps_ratio'] += ( sum(map(str.isupper, re.split('[ -]', tweet))) - nonword_count ) / (word_count - nonword_count)/N
				authors[author].nonlinguistic_features['titlecase_ratio'] += ( sum(map(str.istitle, re.split('[ -]', tweet))) - nonword_count ) / (word_count - nonword_count)/N

			if ((sum(map(str.isupper, tweet.split(' '))) - nonword_count ) > 0):
				authors[author].nonlinguistic_features['allcaps_inclusion_ratio'] += 1/N

	return (authors)

def extract_POS_features(authors):
	'''
	Extract the features stored in the POS_count dict attribute on the author class. Also updates the POS_tags
	as has previously been extracted (as was done incorrectly).
	Takes dictionary of authors as an input and returns the modified version.
	'''
	print("This may take some time.")
	nlp = spacy.load("en_core_web_md")
	print("Language model imported.")
	for author in authors.keys():

		# First we remake the POS tags list as was incorrectly saved.
		cleaned = __clean_tweets__(authors[author].tweets)
		authors[author].POS_tags = []
		authors[author].tokens = []
		for tweet in nlp.pipe(cleaned, disable=['parser']):
		    #Collect and save tags
		    tags = []
		    text = []
		    tags.append([str(token.pos_) for token in tweet])
		    text.append([str(token.text) for token in tweet])
		    authors[author].POS_tags.append(tags[0])
		    authors[author].tokens.append(text[0])

		N=len(authors[author].tweets)

		authors[author].adjectives = {}
		authors[author].POS_counts = {}

		# Now we go through each tweet and each POS tag and make a count of how many times each tag is used
		# If the tag is ADJ, we take the adjective used (using token) and add it to our adjective dictionary.
		# TOKEN refers to all tokens. So we can determine afterwards a percentage based on how long their tweets are (in terms of tokens)
		tag_list = ['ADJ' ,'ADP', 'ADV', 'AUX' , 'CONJ' , 'CCONJ' , 'DET' , 'INTJ' , 'NOUN' , 'NUM' , 'PART' , 'PRON' , 'PROPN' , 'PUNCT' , 'SCONJ' , 'SYM' , 'VERB' ,'X' , 'SPACE', 'TOKEN']
		for tag in tag_list:
		    authors[author].POS_counts['{}_mean'.format(tag)] = 0
		    
		for i_tweet in range(len(authors[author].POS_tags)):
		     for j_tag in range(len(authors[author].POS_tags[i_tweet])):
		            authors[author].POS_counts['{}_mean'.format(authors[author].POS_tags[i_tweet][j_tag])] += 1/N #We are taking the running average by dividing by N
		            authors[author].POS_counts['TOKEN_mean'] += 1/N #Take running average of how many tokens usually per tweet

		            #Check if adjective. If so add to our adjectives dictionary
		            if authors[author].POS_tags[i_tweet][j_tag] == 'ADJ':
		                if authors[author].tokens[i_tweet][j_tag].lower() in authors[author].adjectives.keys():
		                    authors[author].adjectives[authors[author].tokens[i_tweet][j_tag].lower()] += 1
		                else:
		                    authors[author].adjectives[authors[author].tokens[i_tweet][j_tag].lower()] = 1

		# This is useless and affected a lot by our preprocessing, so I'm removing it so it doesn't affect our model in a dumb way.
		authors[author].POS_counts.pop('SPACE_mean')
		print("|",end='')
	return authors

def extract_lexical_features(Authors):
    # On raw text, get average grade level of the tweets
    for author in Authors.keys():
        Authors[author].readability = 0
        for tweet in Authors[author].tweets:
        	Authors[author].readability += (textstat.text_standard(tweet, float_output=True)/len(Authors[author].tweets))
    
    # On lemmatized text, get the TTR to determine the lexical diversity
    for author in Authors.keys():
        Authors[author].TTR = ld.ttr(Authors[author].clean)

    return Authors

def get_emotion_features(authors):
	word_lexicon = pd.read_csv('data/external/NRC_EmoWord.txt')
		#word_lexicon["aback"], word_lexicon["anger"],word_lexicon["0"] = word_lexicon["aback anger 0"].str.split("\\t", n = 2, expand = True) 

	# new data frame with split value columns 
	new = word_lexicon["aback	anger	0"].str.split("\\t", n = 2, expand = True) 
	  
	# making separate first name column from new data frame 
	word_lexicon["Word"]= new[0] 
	  
	# making separate last name column from new data frame 
	word_lexicon["Emotion"]= new[1] 

	word_lexicon["Value"]= new[2] 
	  
	# Dropping old Name columns 
	word_lexicon.drop(columns =["aback	anger	0"], inplace = True) 


	numpy_lexicon = word_lexicon.to_numpy()


	shorter_lexicon = np.delete(numpy_lexicon, np.where(numpy_lexicon == '0'), axis = 0)
	#short_lex = numpy_lexicon[np.all(numpy_lexicon != 0, axis=1)]

	emotion_dict = {}

	for i, row in enumerate(shorter_lexicon):
	    if row[0] not in emotion_dict.keys():
	        emotion_dict[row[0]] = [f"{row[1]}"]
	    else:
	        emotion_dict[row[0]] += [f"{row[1]}"]
    for author in authors.keys():
        auth_dict = {"anger": 0, "fear": 0, "anticipation": 0, "trust": 0, "surprise": 0, "sadness": 0, "joy": 0,
                 "disgust": 0, "positive": 0, "negative": 0}
        for tweets in authors[author].clean: # Iterate through tokens

            tokens = tweets.split(" ")
            for token in tokens:

                if token in emotion_dict.keys():
                    tags = emotion_dict[token]
                    for i in tags:
                        val = auth_dict[i]
                        auth_dict[i] = val + 1
        authors[author].emotion = auth_dict
    return authors