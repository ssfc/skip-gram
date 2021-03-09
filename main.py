import numpy as np
import torch


# 1. Corpus;
def tokenize_corpus(thisCorpus):
    tokens = [x.split() for x in thisCorpus]

    return tokens

# 2. Creating vocabulary;
def tokenize_corpus(thisCorpus):
    tokens = [x.split() for x in thisCorpus]

    return tokens

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


# 1. Corpus;
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
tokenized_corpus = tokenize_corpus(corpus)
print("Tokenized corpus: ", tokenized_corpus)

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print("Vocabulary: ", vocabulary)

# 3. generate pairs center word, context word
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, treated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make sure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
print("idx_pair: ", idx_pairs)




