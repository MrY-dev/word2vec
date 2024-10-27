import datautils
import numpy as np
import lstm 
import torch
import skipgram
from skipgram import FNN

train_file = datautils.train_file
test_file = datautils.test_file

def one_hot(ind):
    x = torch.zeros(4)
    x[int(ind) -1] = 1
    return x.view(4,1)

def to_tensor(row_list,outputs,embedding_map):
    print(len(row_list),len(outputs))
    X = []
    y = []
    for i,j in zip(row_list,outputs):
        tmp = []
        for word in i:
            if word in embedding_map:
                tmp.append(embedding_map[word])
            else:
                tmp.append(embedding_map['<unk>'])
        X.append(torch.stack(tmp))
        y.append(one_hot(j))
    return X,y

def main():
    train_data = datautils.data_obj(train_file,limit=20000)
    embedding_map = skipgram.generate_embedding(train_data)

    # training
    X_train,y_train = to_tensor(train_data.row_list,train_data.outputs,embedding_map)
    model = lstm.lstm(embedding_dim=128,hidden_dim=512,output_dim=4)
    lstm.train(X_train,y_train,model)

    # testing
    test = datautils.data_obj(test_file)
    X_test,y_test = to_tensor(test.row_list,test.outputs,embedding_map)
    print(lstm.test(X_test,y_test,model))
    torch.save(model,'skipgram-classification-model.pt')

if __name__ == '__main__':
    main()
