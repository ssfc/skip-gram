import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_sentence(sentences):  # split each sentence into list, made up with words;
    tokens = [x.split("*") for x in sentences]

    return tokens


#判断是否有GPU
USE_CUDA = torch.cuda.is_available()

#固定随机种子，以防止多次训练结果不一致
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)

#设定超参数（hyper parameters）
C= 3 #周围单词个数（context window）
K = 100 #下采样（number of negative samoles）
NUM_EPOCHS =  2 #迭代次数
MAX_VOCAB_SIZE = 30000 #训练词向量的单词数
BATCH_SIZE = 64 #批样本数
LEARNING_RATE = 0.2 #学习率
EMBEDDING_SIZE = 100 #词向量长度
LOG_FILE = "word-embedding.log"


#读取数据
with open("text8","r") as fin:
    text = fin.read()

#构造词频字典
text = text.split()

with open("corpus.txt", "r", encoding='UTF-8') as f:
    data = f.readlines()

sentences = []
for line in data:
    sentences.append(line[1: len(line) - 2])

print(sentences)
tokenized_sentence = tokenize_sentence(sentences)  # split each sentence into list, made up with words;
print("Tokenized sentence: ", tokenized_sentence)

words = []
for item in tokenized_sentence:
    words = words + item
text = words


print(text)

vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))

#建立词和索引的对应
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}

#计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)


# DataLoader自动帮忙生成batch
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()  # 解码为词表中的索引
        self.word_to_idx = word_to_idx  # 词：索引 的键值对
        self.idx_to_word = idx_to_word  # 词（列表）
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率
        self.word_counts = torch.Tensor(word_counts)  # 词个数

    def __len__(self):
        # 共有多少个item
        return len(self.text_encoded)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.text_encoded[idx]  # 找到中心词
        pos_indices = list(range(idx - C, idx)) + list((idx + 1, idx + C + 1))  # 中心词前后各C个词作为正样本
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 取余,以防超过文档范围
        pos_words = self.text_encoded[pos_indices]  # 周围单词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)


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
        # pos_labels:[batch_size, (windows_size * 2)]
        # neg_labels:[batch_size, (windows_size * 2 * K)]

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (windows_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (windows_size * 2 * K), embed_size]

        # 向量乘法
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, (windows_size * 2)]z只保留前两维
        neg_dot = torch.bmm(neg_embedding, input_embedding).squeeze(2)  # [batch_size, (windows_size * 2 * K)]z只保留前两维

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 按照公式计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = -(log_pos + log_neg)  # [batch_size]

        return loss

    def input_embeddings(self):
        ##取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()





# 定义一个模型并移动到GPU,forward中已经定义了loss，只需要再给出一个opremazer
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

# 定义优化器：SGD随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

            input_labels = input_labels.long()  # 全部转为LongTensor
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()

            optimizer.zero_grad()  # 梯度归零
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("epoch", epoch, "loss", loss.item())

        embedding_weights = model.input_embeddings()
        np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

'''
model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))

#计算emedding权重矩阵
def evaluate(filename, embedding_weights):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity

#寻找语义相似的单词
def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]

oembedding_weights = model.input_embeddings()
print("simlex-999", evaluate("simlex-999.txt", embedding_weights))
print("men", evaluate("men.txt", embedding_weights))
print("wordsim353", evaluate("wordsim353.csv", embedding_weights))

for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word))

man_idx = word_to_idx["man"]
king_idx = word_to_idx["king"]
woman_idx = word_to_idx["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(idx_to_word[i])
'''