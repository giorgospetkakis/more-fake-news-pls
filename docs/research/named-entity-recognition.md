# Named entity recognition
## Introduction
Named entity recognition is an information extraction step which seeks to preserve meaningful groupings of words that would otherwise lose substantial amounts of the information they carry if they were separated during processing ("Donald Trump", "White House", etc.).

## Implementations
Luckily, there's a number of modules and libraries available in Python to perform NER.
### [spaCy](https://spacy.io/)
- Open source software library for (advanced) NLP
- Includes not only NER, but many other useful features (POS tagging, word embeddings, tokenization, etc.)

For NER, they have [models](https://spacy.io/api/annotation#named-entities) trained on different corpora, so depending on which one is chosen, how fine-grained the tags are varies. I think the set that would be the most relevant for us are the models trained on the OntoNotes 5 corpus. The supported entities for those models are shown in the image below. The Wikipedia corpus models might also be useful, though.

![Image of supported entities](https://miro.medium.com/max/1400/1*qQggIPMugLcy-ndJ8X_aAA.png)
### [ner-d](https://pypi.org/project/ner-d/)
- Python module specifically for NER
## How can we use it?
I for sure think we should implement this in some form. Many of our tweets will/are without a doubt political and being able to track which entities a given user talks about might at the very least give us an indication of their interest/focus.

## Literature
