import nltk
import csv
import urllib.request
import pandas as pd
import numpy as np

testText = open('/Users/sarahletchford/Downloads/artNYT.txt', 'r').read()
tokensentences = [nltk.sent_tokenize(testText)]

resultfile = open('four.csv', 'w')
wr = csv.writer(resultfile, dialect='excel')
for item in tokensentences:
        wr.writerow(item)

a = zip(*csv.reader(open('four.csv', 'r')))
csv.writer(open('finaloutput.csv', 'w')).writerows(a)

test_data = 'finaloutput.csv'

# Preparing data sets from online sources
url = 'https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH=SI650/training.txt.'

with urllib.request.urlopen(url) as response, open('train_data.csv', 'w') as out_file:
    data = response.read()
    text = data.decode('utf-8')
    out_file.write(text)

train_data_file_name = 'train_data.csv'

test_data_df = pd.read_csv(test_data, header=0, delimeter="\t", quoting=3)
test_data_df.columns = ['Text']

train_data_df = pd.read_csv(train_data_file_name, header=0, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment", "Text"]

# shape of data

train_data_df.Sentiment.value_counts()
np.mean([len(s.split(" ")) for s in train_data_df.Text])

# prepare corpus and build classifier

import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
        return stemmed

def tokenize(Text):
    text = re.sub('[^a-zA-Z]', ' ', Text)
    tokens = nltk.word_tokenize(Text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = 'tokenize',
    lowercase = True,
    stop_words = 'english',
    max_features = 85

)

corput_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corput_data_features_nd = corput_data_features.toarray()

from sklean.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    corpus_data_features_nd[0:len(train_data_df)],
    train_data_df.Sentiment,
    train_size=0.85,
    random_state=1234

)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(x=x_train, y=y_train)

y_pred = log_model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

log_model = LogisticRegression()
log_model = log_model.fit(x=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)

test_pred = log_model.predict(corpus_data_features_nd[len(train_data_df):])

import random
spl = random.sample(range(len(test_pred)), 15)

for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print(sentiment, text)
