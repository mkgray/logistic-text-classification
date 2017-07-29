# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:01:25 2017

@author: Matthew
"""

# IMPORTS
import numpy as np
import pandas as pd
import operator
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# CUSTOM TOKENIZER FOR DATA
def sms_tokenizer(text):
    no_number_text = re.sub(r'\d+', ' ', text)              # Strip digits
    lowercase_text = no_number_text.lower()                 # Lowercase
    alphanumeric_text = re.sub(r'\W+', ' ', lowercase_text) # Strip special chars    
    tokens = alphanumeric_text.split()
    long_tokens = [token for token in tokens if len(token)>2]
    return long_tokens

# DATA INPUTS
dataset_path = 'SMSSpamCollection'
column_names = ['label', 'text']

# START OF PROGRAM
data = pd.read_csv(dataset_path, sep='\t', header=None)
data.columns = column_names

# FEATURE EXTRACTION METHOD
vectorizer = CountVectorizer(min_df=1, tokenizer=sms_tokenizer)
X = vectorizer.fit_transform(data['text'])
y = np.array(data['label'].apply(lambda x: 1 if x=='spam' else 0).tolist())

# CLASSIFIER
classifier = LogisticRegression()
classifier.fit(X, y)

# MATCH THE LOGISTIC REGRESSION COEFFICIENTS TO THE ORIGINAL WORDS
word_features = vectorizer.get_feature_names()
coefficients = classifier.coef_[0]
logistic_weight_mapping = dict(zip(word_features, coefficients))
sorted_x = sorted(logistic_weight_mapping.items(), key=operator.itemgetter(1))

# OUTPUT MOST TELLING GRAMS
print("Top 10 most telling grams of ham:")
print(sorted_x[:10])

print("Top 10 most telling grams of spam:")
print(sorted_x[-10:])






