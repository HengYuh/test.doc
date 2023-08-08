import torch.nn

from vocab import Vocab
from torch import nn
import numpy as np


#def read_pretrained_wordvec(path,vocab:Vocab,word_dim):
#给vocab中的每个词分配词向量，如果有预先传入的训练好的词向量，则提取出来
#path：词向量存储路径
#vocab：词典
#word——dim：词向量的维度
#返回值是词典（按照序号）对应的词向量
    # vecs=np.random.normal(0.0,0.9,[len(vocab),word_dim])    #先随机给词典中的每个词分一个随机词向量
    # with open(path,'r',encoding='utf-8') as file:
    #     for line in file:
    #         line=line.split()
    #         if line[0] in vocab.vocab:      #在词典中提取出来，存到序号对应的一行去
    #             vecs[vocab.word2seq(line[0])]=np.asarray(line[1:],dtype='float32')
    # return vecs


class MyLSTM(nn.Module):
    def __init__(self,vocab_size,word_dim,num_layer,hiden_size,label_num,bidirectional)->None:
        super(MyLSTM,self).__init__()
        #随机生成词向量，随着训练动态更改
        self.embedding_layer=nn.Embedding(vocab_size,word_dim)     #有了词典大小个，dim维的词向量
        self.embedding_layer.weight.requires_grad=True          #实现动态更改
        #假设已经有了大词向量表，不动态更改
        #self.embeddding_layer=nn.Embedding.from_pretrained(torch.from_numpy(vecs).float())
        #self.embedding_layer.weight.requires_grad = False

        self.rnn=nn.LSTM(word_dim,hiden_size,num_layer)

        self.fc=nn.Sequential(nn.Dropout(0.5),
                              nn.Linear(hiden_size,label_num))#可调
#反向传播已经写好了，但前向需要自己写
    def forward(self,X):
        #[seq,batch,word_size]
        X=X.permute(1,0)        #可以将tensor的维度换位，参数表示原来的维度下表
        X=self.embedding_layer(X)  #建立词向量层
        outs,_ =self.rnn(X)         #先喂给lstm
        logits = self.fc(outs[-1])      #lstm的输出中的最后一个cell的输出喂给全连接层做预测
        return logits
