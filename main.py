
# 1. Corpus;
def tokenize_corpus(thisCorpus):
    tokens = [x.split() for x in thisCorpus]

    return tokens

# 2. Creating vocabulary;
def tokenize_corpus(thisCorpus):
    tokens = [x.split() for x in thisCorpus]

    return tokens


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
print(tokenized_corpus)

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print("Vocabulary: ", vocabulary)


