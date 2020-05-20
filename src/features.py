## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import spacy
import numpy as np

def process_semantic_similarity(authors):
	print("Acquiring data")
	nlp = spacy.load("en_core_web_md")
	print("Beginning to extract author semantic similarity. This may take some time.")

	for author in authors.keys():

		# Prepare our array to hold the similarity values for each author
		# Initialize values as -1 so that we can easily separate out 'empty' values later
		a = np.ones((100,100))
		a *= a*-1
		i = 0

		# Pipe to go through the documents faster
		for tweet in nlp.pipe(authors[author].tweets):
		    j = i+1
		    if i == 99:
		        break
		    for compare in nlp.pipe(authors[author].tweets[i+1:]):
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
	print("Done.")
	return(authors)