from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import re
import random
from matplotlib import pyplot as plt
import data
import utils

def run_clustering_NER():

    utils.go_to_project_root()
    authors = data.get_processed_data()

    truths = [auth.truth for auth in list(authors.values())]
    entities = [auth.ents for auth in list(authors.values())]

    for i in range(len(entities)):
        entry = entities[i]
        entities[i] = [e[0] for e in entry if len(e) > 0]

    tweets = []
    for i, e in enumerate(entities):
        tweets += [" ".join([re.sub(r" ", "_", x[0]) for x in e])]

    list_of_sets = []

    split = 200
    tweetsTrain = tweets[:split]
    tweetsTest = tweets[split:]
    true_k = 3

    print("Searching for fake-news spreaders...")
    notif = False

    if sys.argv[1] == None:
        iters = 50
    else:
        iters = int(sys.argv[1])

    while len(list_of_sets) < iters:

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(tweetsTrain)
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

        max_purity = 0 ; pure_cluster = -1
        for c in range(true_k):
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

        if max_purity >= 1.0:
            list_of_sets += [set()]
            for ind in order_centroids[pure_cluster,:100]:
                list_of_sets[-1].add(terms[ind])

    plt.hist(res)
    plt.title(f"Distribution for {true_k} clusters")
    plt.ylabel("Count")
    plt.xlabel("Cluster")

    final_set = list_of_sets[0]
    for s in list_of_sets:
        final_set = final_set.intersection(s)

    file = open("data/interim/fake-news-most-common-NER.txt", "w")
    for term in list(final_set):
        file.writelines(term + "\n")
    file.close()

if __name__ == "__main__":
    run_clustering_NER()