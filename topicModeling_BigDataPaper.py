
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import sys
import io

import gensim
from gensim import corpora
from textutils import DataLoading
from wordcloud import WordCloud
import numpy as np
import pandas as pd

LOADFROMDISK = True

#reload(sys)
#sys.setdefaultencoding('utf8')


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def printInfo(corpus): # prints general information about a corpus of text (list of strings)
    print("*********** GENERAL INFO ON CORPUS **********")
    lengths = [len(i) for i in corpus]
    print("Number of items in the corpus: " + str(len(corpus)))
    print("Avg length of text in the corpus: " + str(float(sum(lengths)) / len(lengths)))
    docs = [doc.split() for doc in corpus]
    print("Total vocabulary of the corpus: " + str(len(corpora.Dictionary(docs))))


def convertToDense(listOfLists, length):
    new2d = []
    for l in listOfLists:
        newList = [0] * length
        for t in l:
            newList[t[0]] = t[1]
        new2d.append(newList)
    return new2d


def show_topics(doc_raw, ldamodel, cleaning = False, combine = False): # Working on new corpora or sub-corpora of different types
    if cleaning :
        doc_clean = [clean(doc).split() for doc in doc_raw]
    else:
        doc_clean = doc_raw
    if (combine):
        doc_clean = [[item for sublist in doc_clean for item in sublist]]
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    print("Working on a test corpus of length " + str(len(corpus)))
    print("****** Topics found in each document of the test corpus ******")
    doc_by_topic = [ldamodel.get_document_topics(bow) for bow in corpus] #, minimum_probability=0.1
    for doc in doc_by_topic:
        print(doc)
    print("***************************************************************")
    topic_densities = convertToDense(doc_by_topic,len(ldamodel.get_topics()))#[[tuple[1] for sublist in doc_by_topic for tuple in sublist]]
    return  topic_densities






# Running and Trainign LDA model on the document term matrix.
if (LOADFROMDISK):
    ldamodel = gensim.models.LdaModel.load("../data/ldaTopicModelRashkinTrainx.model")#ldamodel.load("ldaTopicModelRashkinTrainx")
    dictionary = corpora.Dictionary.load("../data/ldaTopicModelRashkinTrainx.dic")

else:
    # referecne corpus for topic modeling
    rashkin_texts, rashkin_labels = DataLoading.load_data_rashkin("../data/rashkin/xtrain.txt")
    doc_rashkin = [clean(doc).split() for doc in rashkin_texts]

    # either use all test corpora put together or use a reference train corpus
    doc_clean = doc_rashkin  # doc_buzzfeed + doc_buzztop + doc_snopes312

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    print("Dictionary preparation:")
    dictionary = corpora.Dictionary(doc_clean)
    print(len(dictionary))
    dictionary.filter_n_most_frequent(100)
    print(len(dictionary))
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(len(dictionary))
    print("Dictionary finalized.")

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    print(len(doc_term_matrix))

    # Creating the object for LDA model using gensim library
    print("Creating Lda...")
    Lda = gensim.models.ldamodel.LdaModel

    print("Building model...")
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=5, random_state=1)
    ldamodel.save("../data/ldaTopicModelRashkinTrainx.model")
    dictionary.save("../data/ldaTopicModelRashkinTrainx.dic")


# Show topics with some most important words
for i in ldamodel.print_topics(num_topics=10, num_words=10):
    print(i)



# compile other corpora documents
buzzfeed_texts, buzzfeed_labels =  DataLoading.load_data_buzzfeed()
buzztop_texts, buzztop_labels =  DataLoading.load_data_buzzfeedtop()
snopes312_texts, snopes312_labels = DataLoading.load_data_snopes312()
emergent_texts, emergent_labels = DataLoading.load_data_emergent()

doc_buzzfeed = [clean(doc).split() for doc in buzzfeed_texts]
doc_buzztop = [clean(doc).split() for doc in buzztop_texts]
doc_snopes312 = [clean(doc).split() for doc in snopes312_texts]
doc_emergent = [clean(doc).split() for doc in emergent_texts]

#Project into the built topic space

print("For buzzfeed data:")
A = show_topics(doc_buzzfeed, ldamodel, combine=True)

print("For buzztop data:")
B = show_topics(doc_buzztop, ldamodel, combine=True)

print("For snopes312 data:")
C = show_topics(doc_snopes312, ldamodel, combine=True)

print("For emergent data:")
D = show_topics(doc_emergent, ldamodel, combine=True)



A=[]
B=[]
C=[]
D=[]
L=[]

print("For buzzfeed per label:")
unique, counts = np.unique(buzzfeed_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(buzzfeed_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_buzzfeed)[l_index]
    A = A + show_topics(sub_corpus, ldamodel, combine=True)
    L.append("Buzzfeed   " + str(l))


print("For buzztop per label:")
unique, counts = np.unique(buzztop_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(buzztop_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_buzztop)[l_index]
    B= B + show_topics(sub_corpus, ldamodel, combine=True)
    L.append("Buzzfeed   top_fake")# + str(l))


print("For snopes312 per label:")
unique, counts = np.unique(snopes312_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(snopes312_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_snopes312)[l_index]
    C = C + show_topics(sub_corpus, ldamodel, combine=True)
    L.append("Snopes   " + str(l))



print("For emergent per label:")
unique, counts = np.unique(emergent_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(emergent_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_emergent)[l_index]
    D = D + show_topics(sub_corpus, ldamodel, combine=True)
    L.append("Emergent   " + str(l))




#1 VISUALIZATION: topics by wordcloud

from pylab import *

plt.figure(figsize=(30, ldamodel.num_topics))
subplots_adjust(hspace=0.1, wspace=0.1)
plt.axis("off")
for t in range(ldamodel.num_topics):
    ax1 = subplot((ldamodel.num_topics/5 +1), 5, t+1)
    ax1.imshow(WordCloud(background_color="white",max_font_size=70,relative_scaling=0.5,prefer_horizontal=0.7).fit_words(dict(ldamodel.show_topic(t, 10))), interpolation="bilinear")
    ax1.set_title("Topic #" + str(t+1), size = 30)
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
plt.rcParams.update({'axes.titlesize': 'large'})
plt.savefig('test_topics.pdf', format='pdf')



'''

#2 VISUALIZATION: heat map drawing

def heatmap(data, row_labels, col_labels, ax=None,cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """



    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold], size = 0.2)
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
#The above now allows us to keep the actual plot creation pretty compact.




fig, ax = plt.subplots()

topics = ["technology", "mixed", "market", "sports", "health", "personal", "politics", "police", "legislation", "environment"]
datasets = ["unknown",  "mostly true", "mixture", "mostly false", "top fake",
            "true", "mostly true","mixture", "mostly false", "false",
            "unknown", "true", "false"]
densities = np.array(A+B+C+D) #np.array([range(0,10)] * 13) #np.array(A+B+C+D)



im, cbar = heatmap(densities, datasets, topics, ax=ax,
                   cmap="YlGn", cbarlabel="Topic density")

plt.annotate('Buzzfeed', (0,0), (-120, +254), xycoords='axes fraction', textcoords='offset points', va='top', color = "gray")
plt.annotate('Snopes', (0,0), (-120, +154), xycoords='axes fraction', textcoords='offset points', va='top', color = "gray")
plt.annotate('Emergent', (0,0), (-120, +54), xycoords='axes fraction', textcoords='offset points', va='top', color = "gray")


texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
#plt.show()
plt.savefig('test_topic_densities.pdf', format='pdf')

'''


