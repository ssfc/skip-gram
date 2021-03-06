# https://github.com/zhangxiann/Skip-gram/blob/master/word2vec.py
# https://zhuanlan.zhihu.com/p/275899732
# test code found at: https://zhuanlan.zhihu.com/p/105955900

import collections
import os
import random
import zipfile
import numpy as np
import urllib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# 参数设置
EMBEDDING_DIM = 40  # 词向量维度
PRINT_EVERY = 100  # 可视化频率
EPOCHS = 200  # 训练的轮数
BATCH_SIZE = 200  # 每一批训练数据大小
N_SAMPLES = 3  # 负样本大小
WINDOW_SIZE = 3  # 周边词窗口大小
FREQ = 5  # 词汇出现频数的阈值
DELETE_WORDS = False  # 是否删除部分高频词
VOCABULARY_SIZE = 5000
LEARNING_RATE = 0.01


# DataLoader自动帮忙生成batch
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, data, word_freqs):
        super(WordEmbeddingDataset, self).__init__()
        self.data = torch.Tensor(data).long()  # 解码为词表中的索引
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率

    def __len__(self):
        # 共有多少个item
        return len(self.data)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.data[idx]  # 找到中心词
        pos_indices = list(range(idx - WINDOW_SIZE, idx)) + list(
            range(idx + 1, idx + WINDOW_SIZE + 1))  # 中心词前后各C个词作为正样本
        # pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  # 过滤，如果索引超出范围，则丢弃
        pos_indices = [i % len(self.data) for i in pos_indices]
        pos_words = self.data[pos_indices]  # 周围单词
        # 根据 变换后的词频选择 K * 2 * C 个负样本，True 表示可重复采样
        neg_words = torch.multinomial(self.word_freqs, N_SAMPLES * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


# 构造一个神经网络，输入词语，输出词向量
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 模型输出nn.Embedding(30000, 100)
        self.out_embed.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 模型输入nn.Embedding(30000, 100)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        # 权重初始化的一种方法

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels:[batch_size]
        # pos_labels:[batch_size, windows_size*2]
        # neg_labels:[batch_size, windows_size * N_SAMPLES]

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, windows_size * 2, embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (windows_size * 2 * K), embed_size]

        # 向量乘法
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # input_embedding = input_embedding.view(BATCH_SIZE, EMBEDDING_DIM, 1)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, windows_size * 2] 只保留前两维
        neg_dot = torch.bmm(neg_embedding.neg(), input_embedding).squeeze(
            2)  # [batch_size, windows_size * 2 * K] 只保留前两维

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 按照公式计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = -(log_pos + log_neg)  # [batch_size]

        return loss

    def input_embeddings(self):
        ##取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


def tokenize_sentence(sentences):  # split each sentence into list, made up with words;
    tokens = [x.split("*") for x in sentences]

    return tokens


# -------------------------------------- Part 1, prepare dataset --------------------------------------------
start_time = time.time()
with open("corpus.txt", "r", encoding='UTF-8') as f:
    data = f.readlines()

sentences = []
for line in data:
    sentences.append(line[1: len(line) - 2])

print(sentences)
tokenized_sentence = tokenize_sentence(sentences)  # split each sentence into list, made up with words;
print("Tokenized sentence: ", tokenized_sentence)

combined_tokenized_sentence = []
for item in tokenized_sentence:
    combined_tokenized_sentence = combined_tokenized_sentence + item

print(combined_tokenized_sentence)

print('Data size', len(combined_tokenized_sentence))
# print(words)  # it is a list containing all words in sequence;

# 取出频数前 50000 的单词

counts_dict = dict((collections.Counter(combined_tokenized_sentence).most_common(VOCABULARY_SIZE - 1)))
print("counts_dict: ", counts_dict)  # get frequency of each word;

# 建立词和索引的对应
idx_to_word = []
for word in counts_dict.keys():
    idx_to_word.append(word)
print("first: ", idx_to_word)

'''
idx_to_word = []
for word in combined_tokenized_sentence:
    if word not in idx_to_word:
        idx_to_word.append(word)
print("second: ", idx_to_word)
'''

word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

# 把单词列表words转换为编号的列表
data = list()
for word in combined_tokenized_sentence:
    index = word_to_idx[word]
    data.append(index)

print("size of combined_tokenized_sentence: ", len(combined_tokenized_sentence))
print("size of data: ", len(data))

# 计算单词频次
total_count = len(data)
word_freqs = {w: c / total_count for w, c in counts_dict.items()}
print("counts_dict.items(): ", counts_dict.items())
print("word frequency: ", word_freqs)

# 以一定概率去除出现频次高的词汇
if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in data}
    data = [w for w in data if random.random() < (1 - prob_drop[w])]
else:
    data = data

# 计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in counts_dict.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)

# print(word_freqs)

# 构造  dataset 和 data loader
dataset = WordEmbeddingDataset(data, word_freqs)
print("size of dataset: ", dataset.__len__())

data_loader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义一个模型
device = torch.device("cuda")
model = EmbeddingModel(VOCABULARY_SIZE, EMBEDDING_DIM).to(device)

# ----------------------------------- Part 3, construct loss and optimizer --------------------------------------------
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
'''
# ----------------------------------- Part 4, training cycle --------------------------------------------
for epoch in range(EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(data_loader):

        device = torch.device("cuda")

        input_labels = input_labels.long().to(device)  # 全部转为LongTensor
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()  # 梯度归零
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print("epoch", epoch, "i", i, "loss", loss.item())

    embedding_weights = model.input_embeddings()
#    print(embedding_weights)
    np.save("embedding-{}".format(EMBEDDING_DIM), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_DIM))

print("--- %s seconds ---" % (time.time() - start_time))
'''

# 寻找语义相似的单词
'''
def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]

'''
model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_DIM)))
embedding_weights = model.input_embeddings()
print(embedding_weights.shape)
word_first = '23'
print(word_to_idx)
index_first = word_to_idx[word_first]
embedding_first = embedding_weights[index_first]
word_second = '49'
index_second = word_to_idx[word_second]
print(word_to_idx)
embedding_second = embedding_weights[index_second]

print(embedding_first)
print(embedding_second)
print(scipy.spatial.distance.cosine(embedding_first, embedding_second))

