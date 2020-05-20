## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import re
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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

def extract_fake_news_mcts(authors, n_models=50, k=3, threshold=1.0, _max_iter=5000, _n_init=5, data_split=0.6667):
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