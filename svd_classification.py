from svd import svd_embeddings
import datautils
import numpy as np
import lstm 
import torch
import pickle

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
        X.append(torch.Tensor(np.array(tmp)))
        y.append(one_hot(j))
    return X,y

def main():
    svd = svd_embeddings(train_file,limit=50000)
    svd.generate_embeddings(window_size=5,embedding_dim=128)
    embedding_map = svd.get_embeddings()
    torch.save(embedding_map,'svd-word-vectors.pt')
    # training
    X_train,y_train = to_tensor(svd.row_list,svd.outputs,embedding_map)
    model = lstm.lstm(embedding_dim=128,hidden_dim=512,output_dim=4)
    lstm.train(X_train,y_train,model)
    torch.save(model,'svd-classification-model.pt')
    # testing
    test = datautils.data_obj(test_file)
    X_test,y_test = to_tensor(test.row_list,test.outputs,embedding_map)
    print(lstm.test(X_test,y_test,model))

if __name__ == '__main__':
    main()


