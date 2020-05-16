import data
import spacy
import json
from utils import go_to_project_root

def process_tweets():

	'''
	Currently just a function to call with no parameters.
	Returns a dictionary of authors as defined in data.py with spacy-obtained attributes added.
	Will hopefully update so we can save this into a JSON for quicker access. Is somewhat verbose.
	Takes time. Please expect to spend a few minutes processing the data. 
	'''

	print('Loading authors')
	nlp = spacy.load("en_core_web_sm")
	authors = data.get_raw_data('en')
	print('Authors loaded.')

	# We don't want the url and hashtag denoters to be in this data collection
	ignore = ['URL', 'HASHTAG']

	print('Processing author data')

	# So we can save easier later.
	go_to_project_root()
	# Go through each author
	for author in authors.keys():
		# Pipe to go through the documents faster
		for tweet in nlp.pipe(authors[author].tweets, disable=['parser']):

			#Named entity recognition
			authors[author].ents.append(list(tweet.ents))

			#Collect and save tags
			tags = []
			tags.append(token.pos_ for token in tweet if token.text not in ignore)
			authors[author].POS_tags.append(tags)

			# Collect only the lemma form of the words. Only words with only alpha characters kept.
			lemmas = []
			lemmas.append([token.lemma_.lower() for token in tweet if (token.is_alpha and (token.text not in ignore))])
			authors[author].clean.append(" ".join(lemmas[0]))

			# Now save into the JSON file. This doesn't work right now. Will work on it.
			# with open("data/processed/{authors[author].author_id}.json", "w") as file:
			# 	json.dump(authors[author].__dict___, file)

	print('Data processed and saved')
	return(authors)


# # To compare noun chunking vs named entity recognition.
# print([chunk.text for chunk in nlp(authors['06ct0t68y1acizh9eow3g5rhancrppr8'].tweets[9]).noun_chunks])
# print([ent.label_ for ent in nlp(authors['06ct0t68y1acizh9eow3g5rhancrppr8'].tweets[9]).ents])