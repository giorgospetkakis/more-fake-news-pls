from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import re
import random
from matplotlib import pyplot as plt

sys.path += ['/Users/giorgos.petkakis/Documents/GitHub/NLP2/src']

import data
document = data.get_raw_data('en')

list_all = list(document.values())
plt.figure(figsize=(16, 9))

for r in range(10):
    random.shuffle(list_all)

    tweets = ['\n'.join(tw.tweets) for tw in list_all]
    tweets = [re.sub(r"HASHTAG|URL|RT|USER|amp", "", t) for t in tweets]

    split = 200
    tweetsTrain = tweets[:split]
    tweetsTest = tweets[split:]

    k_range = range(2, 13)

    purities = []
    for k in k_range:
        print(f"Number of Clusters = {k}")

        total_average_purity = 0
        iters = 10
        for t in range(iters):
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(tweetsTrain)

            true_k = k
            model = KMeans(n_clusters=true_k, algorithm="full", init='k-means++', max_iter=5000, n_init=12)
            model.fit(X)

            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()

            pred = np.zeros((len(tweetsTest), 2))
            for i, tw in enumerate(tweetsTest):
                X = vectorizer.transform([tw])
                predicted = model.predict(X)
                pred[i] = [int(predicted), list_all[i + split].truth]

            res = [(pred[np.where(pred[:, 1]==k)])[:,0] for k in range(2)]

            average_purity = 0
            for c in range(true_k):
                class_a = len(np.where(res[0]==c)[0])
                class_b = len(np.where(res[1]==c)[0])

                class_sum = class_a + class_b
                if class_sum != 0:
                    purity = abs(0.5 - class_a / class_sum) * 2
                else:
                    purity = 0

                average_purity += purity
            average_purity /= true_k

            print(f"Average Purity: {average_purity}")
            total_average_purity += average_purity
        total_average_purity /= iters
        purities += [total_average_purity]

        print(f"Total Average Purity: {total_average_purity}")
    plt.plot(k_range, purities, label=f"Iteration {r}")

plt.ylabel("Purity")
plt.xlabel("k")
plt.xticks(ticks=k_range)
plt.legend()

plt.savefig(f"Average Purity for K")