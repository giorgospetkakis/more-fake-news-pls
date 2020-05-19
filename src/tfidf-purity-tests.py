from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import re
import random
import itertools
from matplotlib import pyplot as plt
import data

authors = data.get_processed_data()

truths = [auth.truth for auth in list(authors.values())]
entities = [auth.ents for auth in list(authors.values())]

for i in range(len(entities)):
    entry = entities[i]
    entities[i] = [e[0] for e in entry if len(e) > 0]

plt.figure(figsize=(16, 9))

combos = []
_types = ["PERSON", "NORP", "FAC", "ORG", "GPE", "PRODUCT", "EVENT", "LAW"]
for i in range(2, len(_types)):
    cs = list(itertools.combinations(_types, i))
    for c in cs:
        combos += [c]

purities = []
for r in combos:
    tweets = []
    for i, e in enumerate(entities):
        tweets += [" ".join([re.sub(r" ", "_", x[0]) for x in e if x[1] in r])]

    # print(f"Iteration {r}")
    split = 200
    tweetsTrain = tweets[:split]
    tweetsTest = tweets[split:]

    k_range = range(2, 60, 3)

    purities = []
    for k in k_range:
        # print(f"Number of Clusters = {k}")

        total_average_purity = 0
        iters = 1
        for t in range(iters):
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(tweetsTrain)

            true_k = k
            model = KMeans(n_clusters=true_k, algorithm="full", init='k-means++', max_iter=5000, n_init=4)
            model.fit(X)

            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()

            pred = np.zeros((len(tweetsTest), 2))
            for i, tw in enumerate(tweetsTest):
                X = vectorizer.transform([tw])
                predicted = model.predict(X)
                pred[i] = [int(predicted), truths[i + split]]

            res = [(pred[np.where(pred[:, 1]==k)])[:,0] for k in range(2)]

            # for i in range(true_k):
            #     print("Cluster %d:" % i),
            #     for ind in order_centroids[i, :10]:
            #         print("%s" % terms[ind])

            cum_purity = 0
            classes_predicted = true_k
            for c in range(true_k):
                class_a = len(np.where(res[0]==c)[0])
                class_b = len(np.where(res[1]==c)[0])

                class_sum = class_a + class_b
                if class_sum != 0:
                    purity = abs(0.5 - class_a / class_sum) * (class_sum/len(tweetsTest))
                else:
                    purity = 0
                cum_purity += purity

            # print(f"Cumulative Purity: {cum_purity}")
            total_average_purity += cum_purity

        total_average_purity /= iters
        purities += [total_average_purity]

        # print(f"Total Average Purity: {total_average_purity}")
    # plt.plot(k_range, purities, label=f"{r}")
    purities += [(r, purities)]

file = open("purities.csv", "w")
file.writelines(purities)
file.close()


# plt.ylabel("Purity")
# plt.xlabel("k")
# plt.xticks(ticks=k_range)
# plt.legend()

# plt.savefig(f"Average Purity")


