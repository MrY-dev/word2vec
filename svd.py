from typing import Iterable
import numpy as np
from scipy import sparse, linalg , stats
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import datautils

class svd_embeddings(datautils.data_obj):
    def __init__(self,path,limit):
        datautils.data_obj.__init__(self,path,limit=limit)
        dim  = len(self.vocab_ind)
        self.sparse_mat = lil_matrix((dim,dim),dtype=float)
        self.embedding_map = {}

    def generate_matrix(self,window):
        row_list = self.row_list
        vocab_ind = self.vocab_ind
        dim  = len(self.vocab_ind)
        mat =  lil_matrix((dim,dim),dtype=float)

        for listofwords in row_list:
            for ind,word in enumerate(listofwords):
                ix1 = vocab_ind[word]
                for i in range(ind-window, ind+window+1):
                    if i > 0 and i < len(listofwords):
                        ix2 = vocab_ind[listofwords[i]]
                        mat[ix1,ix2] += 1
                        mat[ix2,ix1] += 1
                    elif i < 0 :
                        mat[ix1,vocab_ind['<bos>']] += 1
                        mat[vocab_ind['<bos>'],ix1] += 1
                    else:
                        mat[ix1,vocab_ind['<eos>']] += 1
                        mat[vocab_ind['<eos>'],ix1] += 1
        return mat
    
    def generate_embeddings(self,window_size,embedding_dim):
        self.Matrix = self.generate_matrix(window_size)
        self.sparse_mat = self.Matrix
        u,_,_= svds(self.sparse_mat,k = embedding_dim)
        iter = 0

        # normalization
        u =  (u - np.mean(u,axis=0))/np.std(u,axis=0)

        for word in self.vocab:
            self.embedding_map[word] = u[iter]
            iter += 1
        return u

    def get_embeddings(self):
        return self.embedding_map

def main():
    train_file = datautils.train_file
    print('generating embeddings:')
    svd = svd_embeddings(train_file)
    svd.generate_embeddings(1,512)
    print(svd.get_embeddings())


if __name__ == '__main__':
    main()
