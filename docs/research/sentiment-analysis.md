# Sentiment analysis
## Introduction
Sentiment analysis is the process of classifying emotion in text (usually positive-neutral-negative, but can be more fine-grained). 

## How could it be used?
The main idea behind sentiment analysis for fake news detection seems to be that the sentiment would work as an indicator of a given author's emotion towards the topic they are discussing. As such, it could prove suitable as one feature of a set that might be used to assess the truthfulness behind a claim. However, one of the main arguments for its usefulness for fake news detection is that fake news - especially fake news describing other people - tend to have a blatantly negative attitude (accusations, negative adjectives to describe individuals, etc.)

As such, using it on its own would not make sense as negative sentiment can just as well be expressed towards someone for a variety of reasons that have nothing to do with fake news. If we were to use it, I therefore suggest that it be used as one feature out of a set of features (potentially different weights) that we use to make our model's claim.

## Literature
1. [Brief conference doc titled "Fake News Detection Using Sentiment Analysis" from IEEE - writing is godawful, but it talks about some of the preprocessing steps you could take and methods that could be used as well as an argument for why sentiment analysis might be useful for the task, so I wanted to save it anyway](https://ieeexplore-ieee-org.ep.fjernadgang.kb.dk/document/8844880)
