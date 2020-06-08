# Language Processing 2: Electric Boogaloo

__"Given a user’s tweets, can we predict if they have previously shared fake news?"__

by 
[@giorgospetkakis](https://github.com/giorgospetkakis "Giorgos Petkakis")  [@spaidataiga](https://github.com/spaidataiga "Sara Vera Marjanovic")  [@FloraHK](https://github.com/FloraHK "Flora Haahr Kringelbach")

## Abstract  
This article examines the effects of several features, including non-linguistic, lexical, syntactic, semantic and emotional, in detecting fake news. The dataset we used was the one provided by PAN @ CLEF 2020- 300 authors of 100 tweets each. Using a Gradient Boosting classifier we achieved up to 73% validation accuracy, with the most important features being self-Word2Vec word embeddings and emotion features. We also implemented train-time and validation-time data augmentation, which improved our results by generalizing the word embeddings and improving the reliability of validation accuracy. 


## Framework

> __data__  
`raw`: Raw project data  
`external`: Data from external sources and generated data  
`interim`: Intermediate processed data  
`processed`: Final dataset  


> __src__  
`data.py`: Collect data  
`features.py`: Extraction functions  
`feature-extraction-pipeline.py`: Extract the relevant features  
`modelling.py`: Construct predictive models  
`preprocessing.py`: Convert the data to csv  
`utils.py`: Utility functions


> __notebooks__  
Keep experimental jupyter notebook files here
