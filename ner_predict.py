import argparse
import os
import spacy
from spacy import displacy
import numpy as np
import random
import pickle
import json
import pandas as pd

parser = argparse.ArgumentParser(description='Show NER model performance on a single review')
parser.add_argument('--uid', metavar='uid',type=str, required=False,help='uid of a review')
args = parser.parse_args()
uid = args.uid
model_path = os.path.join("model_cli","model-best")
ner_model = spacy.load(model_path)

if uid is None:
    file = open(os.path.join("resources","test_set.data"),'rb')
    test_data = pickle.load(file)

    review_sample = random.choice(test_data)
    review_sample = review_sample[0]
    print(review_sample)
    doc = ner_model(review_sample)
    displacy.serve(doc,style="ent")

if uid is not None:
    path_reviews = os.path.join("resources", "reviews.json")
    with open(path_reviews) as json_file:
        reviews = json.load(json_file)
    df_reviews = pd.DataFrame(reviews)
    review_sample = df_reviews[df_reviews.uid==uid]["body"].iloc[0]
    print(review_sample)
    doc = ner_model(review_sample)
    displacy.serve(doc,style="ent")