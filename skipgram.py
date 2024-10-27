from pathlib import Path
import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
import multiprocess as mp
import datautils
import pandas as pd
import time



class CustomDataset(Dataset):
    def __init__(self,data:datautils.data_obj,c_win,k):
        my_file = Path('./data.csv')
        if  my_file.is_file() == False:
            self.row_data = data.row_list  
            self.word2idx = data.vocab_ind
            # each word we need +,- samples
            self.inputs = []
            self.outputs = []
            print(len(self.word2idx))

            self.freq = data.freq
            denom = 0
            # get problability distribution
            prob = []
            for _,value in self.freq.items():
                prob.append((value)**(0.75)) 
                denom += (value)**(0.75)
            prob = [ i/denom for i in prob]
            #for row in tqdm(self.row_data):
            print(len(prob))
            print(len(self.freq))

            inputs = self.inputs
            outputs = self.outputs
            word2idx = self.word2idx
            thresh = {}
            def task(row):
                for ind,word in enumerate(row):
                    ind_word = word2idx.get(word,'<unk>')
                    thresh[word] = thresh.get(word,0) + 1
                    for i in range(ind-c_win,ind+c_win+1):
                        if  i > 0 and i < len(row) and i != ind: 
                            with open('data.csv','a') as fd:
                                fd.write(f'{ind_word},{word2idx[row[i]]},{1}\n')
                    neg_words_idx = np.random.choice(range(0,len(prob)),2*c_win*k,p=prob)
                    for i in neg_words_idx:                                                                                                      
                        with open('data.csv','a') as fd:
                            fd.write(f'{ind_word},{i},{-1}\n')


            with mp.Pool(16) as pool:
                for _ in tqdm(pool.imap(task,self.row_data),total=len(self.row_data)):
                    pass

        self.df = pd.read_csv('data.csv',header=None,low_memory=True)
    def __len__(self):
        return len(self.df.iloc[:,0])

    def __getitem__(self,idx):
        return torch.tensor([self.df.iloc[idx,0],self.df.iloc[idx,1]]),self.df.iloc[idx,2]


    
class FNN(nn.Module):
    def __init__(self,embedding_dim,data:datautils.data_obj):
        super().__init__()
        self.embedding_dim = embedding_dim
        input_size = len(data.vocab_ind) + 2
        # layers
        self.word_embedding = nn.Embedding(input_size,embedding_dim)
        self.context_embedding = nn.Embedding(input_size,embedding_dim)

    def forward(self,X):
        w,c = torch.tensor_split(X,2,dim=1)
        return self.word_embedding(w),self.context_embedding(c)

def get_data(limit,flag):
    train_file = datautils.train_file
    data = datautils.data_obj(train_file,limit=limit,flag=flag)
    return data


def train(model,dataloader):
    optimizer =  optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
    model.to('cuda')
    epochs = 10 
    loss_fn = nn.CosineEmbeddingLoss()
    for epoch in range(epochs):
        running_loss = 0
        for X,y in tqdm(dataloader,total=len(dataloader)):
            optimizer.zero_grad()
            X = X.to('cuda')
            y = y.to('cuda')
            in1,in2= model(X)
            in1 = torch.reshape(in1,(-1,128))
            in2 = torch.reshape(in2,(-1,128))
            loss =loss_fn(in1,in2,y)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch [{epoch+1}/{epochs}]  loss is {running_loss/len(dataloader)}')
    return model

def generate_embedding(train_data:datautils.data_obj):
    model = torch.load('skipgram.pt')
    word2idx = train_data.vocab_ind
    print(len(word2idx))
    print(model)
    embedding_map = {}
    model = model.to('cuda')
    for param in model.parameters():
        param.requires_grad = False
    
    for key,value in tqdm(word2idx.items()):
        inp = torch.tensor([value,value]).view(1,2)
        w,_ = model(inp.to('cuda'))
        embedding_map[key] = w
    return embedding_map

def main():
    # generate train dataset
    print('getting data information...')
    train_data = get_data(limit=20000,flag=False)
    print('creating custom dataset...')
    start = time.time()
    trainset = CustomDataset(train_data,c_win=3,k=2)
    print('time take to generate dataset is',time.time() - start)
    print('creating dataloader ....')
    trainloader = DataLoader(trainset,num_workers=8,batch_size=256)
    print('training model......')
    model = FNN(128,train_data) 
    model = train(model,trainloader)
    torch.save(model,'skipgram.pt')

if __name__ == '__main__':
     main()
