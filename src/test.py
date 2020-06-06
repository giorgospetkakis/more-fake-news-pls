import features
import data
import numpy as np
import pandas as pd
import itertools as it
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Phrases, Word2Vec
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

def extract_word_embeddings(authors, c=-1):
    print("Adding word embeddings...")
    # Split sentences
    sentences = [[" ".join(tw) for tw in auth.nosw] for auth in list(authors.values())]

    print("Creating trigrams...")
    # Create bigram model
    bigram_model = Phrases(sentences, min_count=1, threshold=1)
    bigram_sentences = []
    for sentence in sentences:
        bigram_sentence = " ".join(bigram_model[sentence])
        bigram_sentences += [bigram_sentence.split(" ")]

    # Create trigram model
    trigram_model = Phrases(bigram_sentences)
    trigram_sentences = []
    for bigram_sentence in bigram_sentences:
        trigram_sentence = " ".join(trigram_model[bigram_sentence])
        trigram_sentences += [trigram_sentence]

    # Only train a model if we don't have the word embeddings trained already
    if type(c) == int:
        # initiate the model and perform the first epoch of training
        print("Training Word2Vec model...")
        sentences_split = [tr.split(" ") for tr in trigram_sentences]
        model = Word2Vec(sentences_split, size=300, window=4, min_count=1, sg=1, workers=7)
        model.train(sentences_split, total_examples=model.corpus_count, epochs=10) # 10 epochs is good
        ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]
        ordered_vocab = sorted(ordered_vocab, key=lambda p: (lambda x,y,z: (-z))(*p))
        ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
        word_vectors = pd.DataFrame(model.wv.vectors[term_indices, :], index=ordered_terms)
    else:
        word_vectors = c
    
    print("Exporting...")
    for i, a in enumerate(authors.keys()):
        text = trigram_sentences[i].split()
        vec = []
        # Add all the word vectors
        for trigram in text:
            if trigram in word_vectors.index:
                vec += [word_vectors.loc[trigram, :].to_numpy()]

        vec = np.mean(vec, axis=0)
        authors[a].embeddings = vec
        
    return authors, word_vectors

authors = data.get_processed_data()
extract_word_embeddings(authors)
