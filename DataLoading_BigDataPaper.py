import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
#from keras.utils.np_utils import to_categorical as to_cat
import random


def generalfunct():
    print("yes")







def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = str(string)
    #string = re.sub(r"\\", "", string.decode("utf-8"))
    #string = re.sub(r"\'", "", string.decode("utf-8"))
    #string = re.sub(r"\"", "", string.decode("utf-8"))
    string = ''.join(e for e in string if (e.isspace() or e.isalnum()))  # comment the if part for Mehvish parser
    return string.strip().lower()
    '''

    ## using this version for NE recognition and anonymization
    string = re.sub(r"\\", " ", string.decode("utf-8"))
    string = re.sub(r"\'", " ", string.decode("utf-8"))
    #string = re.sub(r"\"", " ", string.decode("utf-8"))
    return string
    '''




def balance_data(texts, labels, sample_size, discard_labels=[]):
    ## sample size is the number of items we want to have from EACH class
    unique, counts = np.unique(labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    all_index = np.empty([0], dtype=int)
    for l, f in zip(unique, counts):
        if (l in discard_labels):
            print ("Discarding items for label " + str(l))
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
    print ("Final size of dataset:")
    unique, counts = np.unique(bal_labels, return_counts=True)
    print (np.asarray((unique, counts)).T)
    print ("Final size of remaining dataset:")
    unique, counts = np.unique(rem_labels, return_counts=True)
    print (np.asarray((unique, counts)).T)
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
    #   print(i, type(data_train.data[i]))
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(str(data_train.label[idx]))
    transdict = {
        '1': 2,  # Satire
        '2': 3,  # Hoax
        '3': 1,  # Propaganda
        '4': 0  # Truested
    }
    labels = [transdict[i] for i in labels]
    #labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def load_data_buzzfeed(file_name="../data/buzzfeed/buzzfeed-v02-originalLabels.txt"):
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
    #labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def load_data_snopes(file_name, classes = 2 ):
    # Useful for reading from the following files:
    # "../data/snopes/snopes_checked_v02_right_forclassificationtest.csv"
    # "../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv"

    print("Loading data...")
    data_train = pd.read_csv(file_name)
    print(data_train.shape)
    print(data_train.label[0:10])
    print(data_train.label.unique())
    # print(data_train[data_train["label"].isnull()])
    texts = data_train.data
    labels = data_train.label

    if (classes == 2):
        transdict = {
            'ftrue': 0,
            'mtrue': 0,
            'mfalse': 1,
            'ffalse': 1,

            'mixture': 5,
        }
    else:
        transdict = {
            'ftrue': 0,
            'mtrue': 1,
            'mixture': 2,
            'mfalse': 3,
            'ffalse': 4,
        }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(labels[0:10])
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
        'true': 1,
        'mostly true': 2,
        'mixture': 3,
        'mostly false':4,
        'false': 5
    }

    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels


def load_data_buzzfeedtop(file_name="../data/buzzfeed-top/buzzfeed-top.csv"):
    print("Loading data...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")
    print(df.shape)
    print(df[0:3])
    texts = df.original_article_text_phase2.apply(lambda x: clean_str(BeautifulSoup(x).encode('ascii', 'ignore')))
    labels = [0] * len(df.original_article_text_phase2) # all are false news
    #labels = "false" * len(df.original_article_text_phase2)
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels




def load_data_emergent(file_name="../data/emergent/url-versions-2015-06-14.csv"):
    print("Loading data...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

    print(df.shape)
    print(df[0:3])
    df = df.drop_duplicates(
        df.columns.difference(['articleVersionId', 'articleVersion', 'articleUrl']),
        keep="last")
    df = df[df["articleBody"] != ""]
    print(pd.crosstab(df["articleStance"], df["claimTruthiness"], margins=True))

    df = df[df["articleStance"] == "for"]

    labels = df.claimTruthiness
    texts = df.articleBody.apply(lambda x: clean_str(BeautifulSoup(str(x)).encode('ascii', 'ignore')))

    transdict = {
        'true': 1,
        'false': 2,
        'unknown': 0
    }

    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels