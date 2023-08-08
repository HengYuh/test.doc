class Vocab():
    def __init__(self, datas:list,limit_size)-> None:
        self.vocab = {}
        self.padding_word='<pad>'
        self.stop_word=[]
        #提取词频
        cnt ={}
        for data in datas:
            for word in data:
                if word in self.stop_word:
                    continue
                if word not in cnt:cnt[word] = 1
                else:cnt[word] += 1

        self.vocab[self.padding_word]=0
        if len(cnt)>limit_size:
            #将词频从词典提取为列表，并按照lamda排序方式顺时针排序
            cnt=sorted(cnt.items(),key=lambda t:t[1],reverse=True)
            for w,_ in cnt:
                if len(self.vocab)==limit_size:break
                self.vocab[w]=len(self.vocab)
        else:
            for w, _ in cnt:
                self.vocab[w] = len(self.vocab)
    #定义len方法，统计词典大小
    def __len__(self):
        return len(self.vocab)#统计词典有多大
    def word2seq(self,word) -> int:
        if word not in self.vocab:
            return self.vocab[self.padding_word]
        return  self.vocab[word]
    #到此为止，词典构造结束

    def set_stopword(self,path=r"C:\Users\heng'ge\Desktop\TextClassify\TextClassification-v1\data\scu_stopwords.txt"):
       with open(path,'r',encoding='utf-8') as fr:
            self.stop_words=[line.strp() for line in fr.readline()]
