import nltk
import re
import gensim

nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
from gensim.models import Word2Vec

neg_review_tr = movie_reviews.sents(list(movie_reviews.fileids('neg'))[0:900])
pos_review_tr = movie_reviews.sents(list(movie_reviews.fileids('pos'))[0:900])

neg_review_te = movie_reviews.sents(list(movie_reviews.fileids('neg'))[900:1000])
pos_review_te = movie_reviews.sents(list(movie_reviews.fileids('pos'))[900:1000])

neg_review_train = [[] for i in range(len(neg_review_tr))]
pos_review_train = [[] for i in range(len(pos_review_tr))]

neg_review_test = [[] for i in range(len(neg_review_te))]
pos_review_test = [[] for i in range(len(pos_review_te))]

#DATA SETUP:
#training data
print("Setting up training data...")
for i in range(len(neg_review_tr)):
    for j in range(len(neg_review_tr[i])):
        match = re.match('[A-Za-z]', neg_review_tr[i][j])
        if match:
            neg_review_train[i].append(neg_review_tr[i][j].lower())

for i in range(len(pos_review_tr)):
    for j in range(len(pos_review_tr[i])):
        match = re.match('[A-Za-z]', pos_review_tr[i][j])
        if match:
            pos_review_train[i].append(pos_review_tr[i][j].lower())
            

#testing data
for i in range(len(neg_review_te)):
    for j in range(len(neg_review_te[i])):
        match = re.match('[A-Za-z]', neg_review_te[i][j])
        if match:
            neg_review_test[i].append(neg_review_te[i][j].lower())

for i in range(len(pos_review_te)):
    for j in range(len(pos_review_te[i])):
        match = re.match('[A-Za-z]', pos_review_te[i][j])
        if match:
            pos_review_test[i].append(pos_review_te[i][j].lower())

#combining pos and neg training data sets for model
training_data = []
for i in range(len(neg_review_train)):
    training_data.append(neg_review_train[i])

for i in range(len(pos_review_train)):
    training_data.append(pos_review_train[i])

#combining pos and neg testing data sets
testing_data = []
for i in range(len(neg_review_test)):
    testing_data.append(neg_review_test[i])

for i in range(len(pos_review_test)):
    testing_data.append(pos_review_test[i])


#WORD2VEC MODEL SETUP:
#setting up gensim Word2Vec model using training data
model = gensim.models.Word2Vec(training_data)
terms = list(model.wv.index_to_key)
print("Total of ", end = "")
print(len(terms), end = " ")
print("terms in the Word2Vec model")


