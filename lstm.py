import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class lstm(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,output_dim):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,row_embedding): # embedding ofa row in train.csv,test.csv
        lstm_out,_ = self.lstm(row_embedding.view(len(row_embedding),1,-1))
        desc = self.fc(lstm_out.view(len(row_embedding),1,-1))
        return desc

device = 'cuda'

def train(X,y,model,epochs=20):  # X->  list of row embeddings which are tensors, y -> class
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0

        for input,output in tqdm(zip(X,y),total=len(X)):
            input = input.to(device)
            output = output.to(device)
            model.zero_grad()
            pred = model(input)
            tmp = torch.mean(pred,0)
            loss = loss_fn(tmp.flatten(),output.flatten())
            loss.backward()
            optimizer.step()
            running_loss += loss

        print(f'epoch {epoch+1},loss: {running_loss/len(X)}')
    return model

def test(X_test,y_test,model): # X->  list of row embeddings which are tensors, y -> class
    model.to(device)
    model.eval()
    acc = 0
    tot = 0
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    with torch.no_grad():
        for input,output in zip(X_test,y_test):
            input = input.to(device)
            output = output.to(device)
            pred = model(input)
            tmp = torch.mean(pred,0)
            if output.flatten().argmax() == tmp.flatten().argmax():
                acc += 1
            tot += 1
            loss += loss_fn(tmp.flatten(),output.flatten())
    return acc/tot,loss/tot
    
