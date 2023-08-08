from numpy import dtype
from torch.utils.data import Dataset
import torch
import jieba            #中文，可更换提升效率
from tqdm import tqdm  #进度条，可以看进行到什么程度

#原始数据的读入
def read_cnews_data(path):
    labels=[]
    inputs=[]
    label_num={}
    with open(path,'r',encoding='utf-8') as file:
        for i, line in enumerate(file):#按行读取
            line=line.strip()
            sample = line.split('\t')
            #inputs.append(sample[0])
            #labels.append(sample[1])      1.这里注释掉
            if sample[1] not in label_num:           #把标签和句子区分开来，line。split（‘《sep》’）变成list
                label_num[sample[1]]=len(label_num)
            labels.append(int(sample[1]))
            inputs.append(sample[0])
    print(label_num)
    return inputs,labels,len(label_num)

class CnewsDataset(Dataset):
    def __init__(self,path)-> None:
        super().__init__()
        self.inputs, self.labels, self.label_num = read_cnews_data(path)
        self.data2token()
    def data2token(self):
        self.avg_len=0
        for i,data in enumerate(tqdm(self.inputs)):
            self.inputs[i]=jieba.lcut(data)#jieba分词
            self.avg_len+=len(self.inputs[i])  # 2.这里加[i]
        self.avg_len/=len(self.labels)
        print(f'the average len is{self.avg_len}')
    #将词转为序列,用padding短句填充，长句截断
    def token2seq(self,vocab,padding_len):
        for i,data in enumerate(self.inputs):
            if len(self.inputs[i])<padding_len:
                self.inputs[i]+=[vocab.padding_word]*(padding_len-len(self.inputs[i]))
            elif len(self.inputs[i])>padding_len:
                self.inputs[i]=self.inputs[i][:padding_len]
            for j in range(padding_len):
                self.inputs[i][j]=vocab.word2seq(self.inputs[i][j])
            #将数据转为pytorch中要求的数据类型
            self.inputs[i]=torch.tensor(self.inputs[i],dtype=torch.long)
            #self.inputs[i]=torch.tensor(self.inputs[i], dtype=torch.long)
        #self.labels = [int(label) for label in self.labels]
        self.labels=torch.tensor(self.labels,dtype=torch.long)
        #print(type(self.inputs[i]))
#每次继承后，如下的两个方法都必须复写，
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,item:int):
        return self.inputs[item],self.labels[item]

# if __name__=='__main__':
#     from vocab import vocab
#     train_inputset = CnewsDataset(r'../data/train.txt')
#     vocab=vocab(train_inputset.inputs,5000)
#     print(vocab.word2seq('中华'))


