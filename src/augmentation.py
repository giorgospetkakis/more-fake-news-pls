import data
import random

authors = data.get_raw_data()
ones = [author for author in authors.values() if author.truth == 1]
zeros = [author for author in authors.values() if author.truth == 0]

tweets_1 = []
tweets_0 = []

for z in zeros:
    for tweet in z.tweets:
        tweets_0 += [tweet]
random.shuffle(tweets_0)

for o in ones:
    for tweet in o.tweets:
        tweets_1 += [tweet]
random.shuffle(tweets_1)

for author in zeros:
    authors[author.author_id].tweets = []
    for i in range(100):
        authors[author.author_id].tweets += [tweets_0.pop(0)]

for author in ones:
    authors[author.author_id].tweets = []
    for i in range(100):
        authors[author.author_id].tweets += [tweets_1.pop(0)]

for i, author in enumerate(authors.keys()):
    authors[author].author_id = f"shuffled-{i + 1}"

for author in authors.values():
    data.exportJSON(author, "data/external/augmented/")