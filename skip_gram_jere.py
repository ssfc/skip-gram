# https://zhuanlan.zhihu.com/p/164779764

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(1224)
np.random.seed(1224)
torch.manual_seed(1224)

if USE_CUDA:
    torch.cuda.manual_seed(1224)

# 设定hyper parameters
C = 3  # context window, namely nearby words threshold
K = 100  # number of negative samples, 每出现一个正确的词就要出现100个错误的词
NUM_EPOCHS = 1
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100
LOG_FILE = 'word_embedding.log'


# Preprocessing
def word_tokenize(text):
    return text.split()


with open(file='text8', mode='r') as fin:
    text = fin.read()

# 构建词汇表
text = [word for word in word_tokenize(text=text.lower())]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 重新normalized一次
VOCAB_SIZE = len(idx_to_word)

"""
为了使用Dataloader，我们需要定义以下两个function:
    - __len__(), 需要返回整个数据集中有多少item
    - __getitem__(), 根据给定的index返回一个item
有了Dataloader后，可以轻松随机打乱整个数据集，拿到一个batch的数据等。
"""


# DataLoader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<unk>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):  # 返回数据用于训练
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], replacement=True)
        return center_word, pos_words, neg_words


# 创建dataset和dataloader
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataLoader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


# 定义PyTorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size)
        self.out_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_labels, pos_labels, neg_lables):  # loss function
        """
        :param input_labels: [batch_size]
        :param pos_labels: [batch_size, (window_size * 2)]
        :param neg_lables: [batch_size, (window_size * 2 * K)]
        :return: loss, [batch_size]
        """
        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_lables)  # [batch_size, (window_size * 2 * K), embed_size]

        # unsqueeze()升维, squeeze()降维
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1], 第二个维度加1
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze()  # [batch_size, (window_size * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze()  # [batch_size, (window_size * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):  # 取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


# 定义一个模型以及把模型移动到GPU
model = EmbeddingModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()


# 评估模型
def evaluate(filename, embedding_weights):
    if filename.endswith('.csv'):
        data = pd.read_csv(filename, sep=',')
    else:
        data = pd.read_csv(filename, sep='\t')
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))
    return scipy.stats.spearmanr(human_similarity, model_similarity)


def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


# 训练模型
"""
1.模型一般需要训练若干个epoch，每个epoch我们都把所有数据分成若干个batch，把每个batch的输入和输出都包装成cuda tensor；
2.forward pass，通过输入的句子预测每个单词的下一个单词，用模型的预测和正确的下一个单词计算cross entropy loss；
3.清空模型当前的Gradient；
4.backward pass，更新模型参数；
5.每隔一定的iteration，输出模型在当前iteration的loss以及在验证数据集上做模型的评估。
"""
optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataLoader):
        # print(input_labels, pos_labels, neg_labels)
        # if i > 2:
        #     break
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()  # 传入参数给forward()函数
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(file=LOG_FILE, mode='a', encoding='UTF-8') as f_out:
                    f_out.write('Epoch: {}, Iteration: {}, Loss: {} + \n'.format(e, i, loss.item()))
                    print(f'Epoch: {e}, Iteration: {i}, Loss: {loss.item()}')


    embedding_weights = model.input_embedding()
    np.save('embedding-{}'.format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), 'embedding-{}.th'.format(EMBEDDING_SIZE))





