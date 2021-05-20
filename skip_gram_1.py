# Step 1: Import libraries

import torch
torch.manual_seed(10)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn import decomposition
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10,8)
import nltk

nltk.download('stopwords')
nltk.download('punkt')

#Import stopwords
from nltk.corpus import stopwords

# Step 2: Consider we have a simplified corpus of words like below.

corpus = [
    'drink milk',
    'drink cold water',
    'drink cold cola',
    'drink juice',
    'drink cola',
    'eat bacon',
    'eat mango',
    'eat cherry',
    'eat apple',
    'juice with sugar',
    'cola with sugar',
    'mango is fruit',
    'apple is fruit',
    'cherry is fruit',
    'Berlin is Germany',
    'Boston is USA',
    'Mercedes from Germany',
    'Mercedes is a car',
    'Ford from USA',
    'Ford is a car'
]

# Step 3: Skip-Gram model tries to predict context given a word.

def create_vocabulary(corpus):
    '''Creates a dictionary with all unique words in corpus with id'''
    vocabulary = {}
    i = 0
    for s in corpus:
        for w in s.split():
            if w not in vocabulary:
                vocabulary[w] = i
                i += 1
    return vocabulary


def prepare_set(corpus, n_gram=1):
    '''Creates a dataset with Input column and Outputs columns for neighboring words.
       The number of neighbors = n_gram*2'''
    columns = ['Input'] + [f'Output{i + 1}' for i in range(n_gram * 2)]
    result = pd.DataFrame(columns=columns)
    for sentence in corpus:
        for i, w in enumerate(sentence.split()):
            inp = [w]
            out = []
            for n in range(1, n_gram + 1):
                # look back
                if (i - n) >= 0:
                    out.append(sentence.split()[i - n])
                else:
                    out.append('<padding>')

                # look forward
                if (i + n) < len(sentence.split()):
                    out.append(sentence.split()[i + n])
                else:
                    out.append('<padding>')
            row = pd.DataFrame([inp + out], columns=columns)
            result = result.append(row, ignore_index=True)
    return result


def prepare_set_ravel(corpus, n_gram=1):
    '''Creates a dataset with Input column and Output column for neighboring words.
       The number of neighbors = n_gram*2'''
    columns = ['Input', 'Output']
    result = pd.DataFrame(columns=columns)
    for sentence in corpus:
        for i, w in enumerate(sentence.split()):
            inp = w
            for n in range(1, n_gram + 1):
                # look back
                if (i - n) >= 0:
                    out = sentence.split()[i - n]
                    row = pd.DataFrame([[inp, out]], columns=columns)
                    result = result.append(row, ignore_index=True)

                # look forward
                if (i + n) < len(sentence.split()):
                    out = sentence.split()[i + n]
                    row = pd.DataFrame([[inp, out]], columns=columns)
                    result = result.append(row, ignore_index=True)
    return result

# Step 4: A bit of preprocessing;

stop_words = set(stopwords.words('english'))

def preprocess(corpus):
    result = []
    for i in corpus:
        out = nltk.word_tokenize(i)
        out = [x.lower() for x in out]
        out = [x for x in out if x not in stop_words]
        result.append(" ". join(out))
    return result

corpus = preprocess(corpus)
corpus


