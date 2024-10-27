import csv
from nltk import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

data_path = './ANLP-2/'


train_file = data_path + 'train.csv'
test_file = data_path + 'test.csv'

class data_obj(object):
    def __init__(self,path,limit=50000,flag = True):
        self.data = []
        self.vocab = []
        self.orig_vocab = []
        self.sent = []
        self.vocab_ind = {}
        self.row_list = []
        self.outputs = []
        self.path = path 
        print(self.path)
        with open(self.path,'r') as f:
            reader = csv.reader(f)
            next(reader,None)
            for i in reader:
                self.data.append(i)

        self.data = self.data[:limit]
    
        self.freq = {}
        freq = self.freq
        #tokenizer = RegexpTokenizer(r'\w+')
        for row in self.data:
            for word in word_tokenize(row[1]):
                word = word.lower()
                freq[word] = freq.get(word,0) + 1 

        for row in self.data:
            tmp = []
            for word in word_tokenize(row[1]):
                word = word.lower()
                self.orig_vocab.append(word)
                if(freq[word] > 1):
                    self.vocab.append(word)
                    tmp.append(word)
                else:
                    del freq[word]
                    self.vocab.append('<unk>')
                    tmp.append('<unk>')
            self.row_list.append(tmp)
            self.outputs.append(row[0])
        # unique words 
        self.vocab = list(set(self.vocab))
        self.orig_vocab = list(set(self.orig_vocab))

        # indexing words
        count = 0
        if flag:
            self.vocab_ind['<bos>'] = count # beginning of sentence tag

        for word in self.vocab:
            count += 1
            self.vocab_ind[word] = count

        if flag:
            self.vocab_ind['<eos>'] = count+1 # end of sentence tag

    def get_data(self):
        return self.data

    def get_vocab(self):
        return self.vocab

    def get_sent(self):
        return self.sent

    def get_map(self):
        return self.vocab_ind

