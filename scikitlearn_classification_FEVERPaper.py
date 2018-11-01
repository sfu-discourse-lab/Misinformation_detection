# Author: Fatemeh Torabi Asr
#         ftasr@github

## Classification code partially taken from: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from bs4 import BeautifulSoup
import re
import pandas as pd

LOAD_DATA_FROM_DISK = True
CLASSES = 5

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,  # default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string.decode("utf-8"))
    string = re.sub(r"\'", "", string.decode("utf-8"))
    string = re.sub(r"\"", "", string.decode("utf-8"))
    return string.strip().lower()


def load_data_liar(file_name):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["id", "label", "data"], usecols=[0, 1, 2])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'true': 1,
        'mostly-true': 2,
        'half-true': 3,
        'barely-true': 4,
        'false': 5,
        'pants-fire': 6
    }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))  #uncomment this for vectorized one-hot representation
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels


def load_data_rubin(file_name="../data/rubin/data.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["label", "data"], usecols=[0, 1])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    # labels = to_cat(np.asarray(labels))
    print(labels[0:6])
    return texts, labels


def load_data_combined(file_name="../data/buzzfeed-debunk-combined/all-v02.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None,
                               names=["id", "url", "label", "data", "domain", "source"], usecols=[2, 3])
    print(data_train.shape)
    print(data_train.label[0:10])
    print(data_train.label.unique())
    # print(data_train[data_train["label"].isnull()])
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'ftrue': 0,
        'mtrue': 1,
        'mixture': 2,
        'mfalse': 3,
        'ffalse': 4,
        'pantsfire': 5,
        'nofact': 6
    }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(labels[0:6])
    return texts, labels


import random


def balance_data(texts, labels, sample_size, discard_labels=[], seed=123):
    np.random.seed(seed)
    ## sample size is the number of items we want to have from EACH class
    unique, counts = np.unique(labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    all_index = np.empty([0], dtype=int)
    for l, f in zip(unique, counts):
        if (l in discard_labels):
            print("Discarding items for label " + str(l))
            continue
        l_index = (np.where(labels == l)[0]).tolist()  ## index of input data with current label
        if (sample_size - f > 0):
            # print "Upsampling ", sample_size - f, " items for class ", l
            x = np.random.choice(f, sample_size - f).tolist()
            l_index = np.append(np.asarray(l_index), np.asarray(l_index)[x])
        else:
            # print "Downsampling ", sample_size , " items for class ", l
            l_index = random.sample(l_index, sample_size)
        all_index = np.append(all_index, l_index)
    bal_labels = np.asarray(labels)[all_index.tolist()]
    bal_texts = np.asarray(texts)[all_index.tolist()]
    remaining = [i for i in range(0, np.sum(counts)) if i not in all_index.tolist()]
    rem_texts = np.asarray(texts)[remaining]
    rem_labels = np.asarray(labels)[remaining]
    print("Final size of dataset:")
    unique, counts = np.unique(bal_labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    print("Final size of remaining dataset:")
    unique, counts = np.unique(rem_labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return bal_texts, bal_labels, rem_texts, rem_labels


def load_data_rashkin(file_name="../data/rashkin/train.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["label", "data"], usecols=[0, 1],
                               dtype={"label": np.str, "data": np.str})
    print(data_train.shape)
    print(data_train[0:6])
    texts = []
    labels = []
    # for i in range(data_train.data.shape[0]):
    #    print(i, type(data_train.data[i]))
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(str(data_train.label[idx]))
    transdict = {
        '1': 3,  # Satire
        '2': 4,  # Hoax
        '3': 2,  # Propaganda
        '4': 1  # Truested
    }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels


def load_data_buzzfeed(file_name="../data/buzzfeed-facebook/bf_fb.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["ID", "URL", "label", "data", "error"],
                               usecols=[2, 3])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'no factual content': 0,
        'mostly true': 1,
        'mixture of true and false': 2,
        'mostly false': 3
    }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels


def load_data_snopes312(file_name="../data/snopes/snopes_checked_v02_forCrowd.csv"):
    print("Loading data...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

    print(df.shape)
    print(df[0:3])
    df = df[df["assessment"] == "right"]
    print(pd.crosstab(df["assessment"], df["fact_rating_phase1"], margins=True))
    labels = df.fact_rating_phase1
    texts = df.original_article_text_phase2.apply(lambda x: clean_str(BeautifulSoup(x).encode('ascii', 'ignore')))
    #
    '''
    texts = []
    labels = []
    print(df.original_article_text_phase2.shape[0])
    print(df.original_article_text_phase2[2])

    for idx in range(df.original_article_text_phase2.shape[0]):
        text = BeautifulSoup(df.original_article_text_phase2[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(df.fact_rating_phase1[idx])
    '''
    transdict = {
        'true': 0,
        'mostly true': 1,
        'mixture': 2,
        'mostly false': 3,
        'false': 4
    }

    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels




#Please contact Rashkin et al. to obtain data for training.
texts_train, labels_train = load_data_rashkin("../data/rashkin/xtrain.txt")
##texts_valid, labels_valid = load_data_liar("../data/liar_dataset/valid.tsv")
#texts_test1, labels_test1 = load_data_rashkin("../data/rashkin/balancedtest.txt")

# Small and Validated Test Datasets:
texts_test1, labels_test1 = load_data_snopes312("../data/snopes/snopes_checked_v02.csv")
#texts_test1, labels_test1 = load_data_combined("../data/buzzfeed/buzzfeed-v02.txt")
#texts_test1, labels_test1 = load_data_rubin()


# texts, labels =  load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")

# texts_test1, labels_test1, texts, labels = balance_data(texts, labels, 200, [6,5])
# texts_valid, labels_valid, texts, labels = balance_data(texts, labels, 200, [6,5])
# texts_train, labels_train, texts, labels = balance_data(texts, labels, 700, [6,5])


'''

if LOAD_DATA_FROM_DISK:
    texts_train = np.load("../dump/trainRaw")
    texts_valid = np.load("../dump/validRaw")
    texts_test1 = np.load("../dump/testRaw")
    labels_train = np.load("../dump/trainlRaw")
    labels_valid = np.load("../dump/validlRaw")
    labels_test1 = np.load("../dump/testlRaw")

    print("Data loaded from disk!")

else:
    texts, labels = load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")
    print("Maximum string length:")
    mylen = np.vectorize(len)
    print(mylen(texts))

    if (CLASSES == 2):
        texts_test1, labels_test1, texts, labels = balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
        texts_valid, labels_valid, texts, labels = balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
        texts_train, labels_train, texts, labels = balance_data(texts, labels, 1400, [2, 3, 4, 5, 6])

    else:
        texts_test1, labels_test1, texts, labels = balance_data(texts, labels, 200, [6, 5])
        texts_valid, labels_valid, texts, labels = balance_data(texts, labels, 200, [6, 5])
        texts_train, labels_train, texts, labels = balance_data(texts, labels, 700, [6, 5])

    texts_train.dump("../dump/trainRaw")
    texts_valid.dump("../dump/validRaw")
    texts_test1.dump("../dump/testRaw")
    labels_train.dump("../dump/trainlRaw")
    labels_valid.dump("../dump/validlRaw")
    labels_test1.dump("../dump/testlRaw")

    print("Data dumped to disk!")
    
    
'''

y_train = labels_train
y_test = labels_test1
target_names, counts = np.unique(y_train, return_counts=True)
print(np.asarray((target_names, counts)).T)

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if (False):  # opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(texts_train)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(1,3))
    X_train = vectorizer.fit_transform(texts_train)
    features = vectorizer.get_feature_names()
    print(features[300000:300200])
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(texts_test1)
print("n_samples: %d, n_features: %d" % X_test.shape)

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    trainScore = metrics.accuracy_score(y_train, clf.predict(X_train))
    score = metrics.accuracy_score(y_test, pred)
    precision = metrics.precision_score(y_test, pred, average='macro')
    recall = metrics.recall_score(y_test, pred, average='macro')
    f1 = metrics.f1_score(y_test, pred, average='macro')
    mse = metrics.mean_squared_error(y_test, pred)

    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if (False):
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    if (True):
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        print(pd.DataFrame({'Predicted': pred, 'Expected': y_test}))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, trainScore, score, precision, recall, f1, mse


results = []

'''
for clf, name in (
        #(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        #(Perceptron(n_iter=50), "Perceptron"),
        #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))
'''

for penalty in ["l2"]:  # , "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

'''
# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

'''

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))

'''
print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

'''

print("=" * 100)
print(results)

'''

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

'''
