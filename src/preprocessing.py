import re
import data
import spacy
import jsonpickle
from utils import go_to_project_root

def process_tweets():
	
	'''
	Currently just a function to call with no parameters.
	Returns a dictionary of authors as defined in data.py with spacy-obtained attributes added.
	Will hopefully update so we can save this into a JSON for quicker access. Is somewhat verbose.
	Takes time. Please expect to spend a few minutes processing the data. 
	'''
	ignore = ['HASHTAG', 'URL']

	print('Loading authors')
	nlp = spacy.load("en_core_web_sm")
	authors = data.get_raw_data('en')
	print('Authors loaded.')

	print('Processing author data')

	# So we can save easier later.
	go_to_project_root()
	# Go through each author
	for author in authors.keys():
		# Pipe to go through the documents faster
		tweets_cleaned = [re.sub(r"HASHTAG|URL|RT|#|https:\/\/t\.\\[a-z\d]+|https.*$|\&*amp", "", tweet) for tweet in authors[author].tweets]
		for tweet in nlp.pipe(tweets_cleaned, disable=['parser']):

			#Named entity recognition
			authors[author].ents.append([(re.sub(r"USER", "", str(ent.text)), str(ent.label_)) for ent in list(tweet.ents) if str(ent.text) != "USER"])

			#Collect and save tags
			tags = []
			tags.append(token.pos_ for token in tweet if token not in ignore)
			authors[author].POS_tags.append(tags)

			# Collect only the lemma form of the words. Only words with only alpha characters kept.
			lemmas = []
			lemmas.append([token.lemma_.lower() for token in tweet if (token.is_alpha and token not in ignore)])
			authors[author].clean.append(" ".join(lemmas[0]))

		__exportJSON__(authors[author])
	print('Data processed and saved')
	return(authors)

def __exportJSON__(author, path='data/processed/'):
	'''
	Exports an author object to a JSON file
	Parameters: 
		author (Author):
			The author to be serialized
	Export: None
	'''
	with open(f"{path}{author.author_id}.json", "w") as file:
		file.writelines(__convert_to_JSON__(author))
		file.close()
	return

def __convert_to_JSON__(author):
	return jsonpickle.encode(author, unpicklable=False)

if __name__ == '__main__':
    process_tweets()

# # To compare noun chunking vs named entity recognition.
# print([chunk.text for chunk in nlp(authors['06ct0t68y1acizh9eow3g5rhancrppr8'].tweets[9]).noun_chunks])
# print([ent.label_ for ent in nlp(authors['06ct0t68y1acizh9eow3g5rhancrppr8'].tweets[9]).ents])