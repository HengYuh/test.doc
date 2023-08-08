import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from data_cnews import CnewsDataset
from vocab import Vocab
from model_news import MyLSTM
from tqdm import tqdm
from numpy import dtype
#device= torch.device('cuda:0')
device=torch.device('cuda'if torch.cuda.is_available() else "cpu")
#以下是预处理与超参数的设定
train_dataset=CnewsDataset(r"C:\Users\heng'ge\Desktop\TextClassify\TextClassification-v1\data\train.txt")
val_dataset=CnewsDataset(r"C:\Users\heng'ge\Desktop\TextClassify\TextClassification-v1\data\train.txt")
test_dataset=CnewsDataset(r"C:\Users\heng'ge\Desktop\TextClassify\TextClassification-v1\data\test.txt")
vocab =Vocab(train_dataset.inputs,5000)      #500个词的词典
train_dataset.token2seq(vocab,16)
val_dataset.token2seq(vocab,16)
test_dataset.token2seq(vocab,16)                  #3.调一下padding len
train_dataset=DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataset=DataLoader(val_dataset,batch_size=64)
test_dataset=DataLoader(test_dataset,batch_size=64)

net=MyLSTM(vocab_size=len(vocab),word_dim=300,num_layer=4,hiden_size=128,label_num=10,bidirectional=True)
#net=MyLSTM(read_pretrained_wordvec(r'glove.6B.50d.txt',vocab,50),len(vocab),50,1,16,10)
net=net.to(device) #网络设置到设备上计算
optimizer=optim.Adam(net.parameters(),lr=1e-3)  #可以换成SGD,lr为学习率
criterion=nn.CrossEntropyLoss().to(device)      #交叉熵损失函数


def train(epoch):
#net.train()和net.eval（）到底什么时候使用，如果一个模型有dropout和batchnormalization，那么它在训练时要以一定概率进行dropout
#但在验证测试时不需要dropout，即net。eval
    def evaluate():
        net.eval()
        correct =0
        all =0
        with torch.no_grad():   #逃避autograd的追踪，因为评估和测试数据不需要计算梯度，也不会进行反向传播
            for(x,y) in tqdm(val_dataset):
                x,y=x.to(device),y.to(device)
                logits= net(x)
                logits=torch.argmax(logits,dim=-1) #从十个概率值找出最大概率值对应的下标
                correct+=torch.sum(logits.eq(y)).float().item()
                all+=y.size()[0]
        print(f'avaluate done! acc{correct/all:5f}')

    for ep in range(epoch):
        print(f'epoch {ep} start')
        net.train()
        for(x,y) in tqdm(train_dataset):
            x,y=x.to(device),y.to(device)
            logits=net(x)       #前向传播求出的预测值，执行net中的forward函数
            loss=criterion(logits,y)    #求loss函数
            optimizer.zero_grad()   #将梯度初始为零（因为训练通常使用mini_batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播函数前面
            loss.backward() #反向传播求梯度，损失函数是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为true，则w的所有上层参数（后面的权重w）的grad会不断更新
            optimizer.step()    #更新所有参数，step（）的作用是进行一次优化步骤，通过梯度下降法来更新参数的值，optimizer只负责通过梯度下降进行优化，而不负责产生梯度
        evaluate()

def test():
        net.eval()
        correct =0
        all =0
        with torch.no_grad():   #逃避autograd的追踪，因为评估和测试数据不需要计算梯度，也不会进行反向传播
            for(x,y) in tqdm(test_dataset):
                x,y=x.to(device),y.to(device)
                logits= net(x)
                logits=torch.argmax(logits,dim=-1) #从十个概率值找出最大概率值对应的下标
                correct+=torch.sum(logits.eq(y)).float().item()
                all+=y.size()[0]
        print(f'test done! acc {correct / all:5f}')

if __name__ == '__main__':
    train(20)
    test()
