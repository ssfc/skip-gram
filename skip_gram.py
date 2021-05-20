# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def tokenize_corpus(this_corpus):  # split each sentence into list, made up with words;
    tokens = [x.split() for x in this_corpus]

    return tokens


# Input layer is just the center word encoded in one-hot manner. It dimensions are [1, vocabulary_size]
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


# 1. Corpus is a list, made up with sentence;
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

print("Corpus: ", corpus)

# 2. Creating vocabulary;
tokenized_corpus = tokenize_corpus(corpus)  # split each sentence into list, made up with words;
print("Tokenized corpus: ", tokenized_corpus)

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)  # add word not in vocabulary into vocabulary;

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}  # create dictionary;
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}  # create dictionary;

vocabulary_size = len(vocabulary)
print("Vocabulary: ", vocabulary)

# 3. generate pairs center word, context word
window_size = 2
idx_pairs = []
# generate idx_pairs for all sentences;
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]  # create list;
    print(indices)  # convert word to numbers(index) representing it;

    # for each word, treated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make sure not jump out sentence
            if (context_word_pos >= 0) and (context_word_pos < len(indices)) and (center_word_pos != context_word_pos):
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))


idx_pairs = np.array(idx_pairs)  # it will be useful to have this as numpy array
# print("idx_pair: ", idx_pairs)

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 200
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()  # x is a vector, size 15;
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)  # 2-dimension times 1-dimensions, return 1 dimension;
        z2 = torch.matmul(W2, z1)  # 2-dimension times 1-dimensions, return 1 dimension;

        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()

    if epo % 10 == 0:
        print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')
