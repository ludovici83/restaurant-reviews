# CONTENTS
-----------

* resources/reviews.json ==> restaurant user reviews bodies and uidentifiers. Dataset for NER and Vectorization.
* resources/entities.json ==> manual labels for NER entities. Contains span position (start, end), type (modifier/concept) 
and review_uid.
* NLP code test EN.pdf ==> code test explanation
* ner_training.ipynb ==> This jupyter notebook contains the training of the Named Entity Recognition model with spacy. This is done in a two-fold way ; on one hand the model is trained using the spacy CLI (command line interface) and saved in the "model_cli" folder, on the other hand a similar model is trained directly by looping over the training epochs in python, this output being saved in the "model_python" folder. In the last part of this notebook, the performance of both trained models is shown by applying the model prediction on a few reviews from a test set of reviews that were held-out during the training phase. For the CLI trained model the precision, recall and f-score metrics are shown.
* ner_predict.py ==> This python script selects a random review from the unseen test dataset and shows the visualized performance of the NER model on the localhost url. In order to observe this, it should be run on a Python IDE (in my case it works fine on PyCharm)
* vectorization_reviews.ipynb ==> This jupyter notebook contains the second part of the test. The reviews in the reviews.json file are preprocessed and then vectorized using tf-idf (term-frequency inverse-document-frequency) method. A cosine-similarity function is provided that computes the similarity between two review vectors. The first 500 reviews are compared using this function and a few examples of similar reviews are shown. 
* model_python ==> spacy NER model trained directly in the jupyter notebook
* model_cli ==> NER model trained using the spacy CLI
* config.cfg ==> spacy config file used for training. Specifies things such as max number of training epochs, the learning rate, the metric used for choosing the best model,etc.
* requirements.txt ==> contains the versions of each python library in my python/anaconda environment (for reproducibility)

Answers to questions

1.1. Which metric have you implemented to evaluate the model? Why?
Answer:
Before training I did a random 60-20-20 split of the initial dataset into training -validation- and test sets. For the model trained via spacy Comand Line Interface (CLI) I used the f-score metric , because it gives equal weight to precision (Percentage of correctly-predicted annotations) and recall (Percentage of reference annotations recovered). The choice of metric for best model selection is done in the config.cfg file in the training.score_weights section (line 109: ents_f=1).

For the model that was trained outside spacy CLI (directly in Python, that is) I checked the performance of the NER model with the unseen reviews of the test dataset (roughly one thousand annotated reviews). In this test set the overall accuracy score of the algorithm in finding the concepts and modifiers was around 93 percent.   

1.2. Does the model have the capacity to find new entities that aren’t present on the training
set? If so, are they from the domain? Otherwise, what would you do to improve that ability?
Answer: In the ner_training.ipynb I show some examples of sentences that are not related to restaurant reviews and that use concepts that are not present in the training set ; such as "el casco de la bicicleta fue de gran calidad", and the model correctly recognizes "casco" and "bicicleta" as "concepts" and "gran" as a modifier, although "casco" and "bicicleta" are not found in the entities.json . Another example was the sentence  "El avión que tenía mi padre cuando era joven era muy rápido", in which the modifiers "joven" and "muy rápido" were recognized but the concept "avión" (again non-existent in the training set) was not recognized by the algorithm. So, the model does have some limited capacity to find new entities that were not present in the training set and that do not belong to the domain. 

2.1. Do vector representations of the sentences hold information about the sequence (order) of words?
Answer: The vectorization that I carried out was based on single token words, so it does not hold info about the order of the words in the sentence. That could be achieved by doing an n-gram tokenization in which the vocabulary for vectorization is not only made up of single words but also by groups of "n" ordered tokens. This however takes a lot more computer memory and execution time.

2.2. If you had more data and time, how would you improve the vectorization?
Answer:
- Well, actually, I would start by doing what is suggested in question 2.1; I would try to do vectorization with 2-grams and 3-grams to see if the reviews that are matched as similar are even better matched than the ones I found. 
- Also; in this test, I used the stemming technique for reducing the size of the overall vocabulary . I would like to try another technique called lemmatization, to see if it gives better results.
- Finally, in this test I used tf-idf vectorizer (based on the term frequency and inverse document frequency of each token in the word-stock). It would be interesting to compare with a word2vec vectorizer. 

