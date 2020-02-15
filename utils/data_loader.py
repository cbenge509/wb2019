import pandas as pd
import numpy as np
import pickle

class DataLoader(object):
    
    def __init__(object):
        pass
    
    def load_dataset(self, dataset_path, indexcol):
        '''return the pickled, pre-processed data'''
        #return pickle.load(open(dataset_path, 'rb'))
        return pd.read_csv(dataset_path, index_col=indexcol)

    def load_clean_words(self, clean_words_path):
        '''return a dict whose key is typo, value is correct word'''
        clean_word_dict = {}
        with open(clean_words_path, 'r', encoding='utf-8') as cl:
            for line in cl:
                line = line.strip('\n')
                typo, correct = line.split(',')
                clean_word_dict[typo] = correct
        return clean_word_dict

    def load_embedding(self, embedding_path):
        '''return a dictionary whose key is word, value is pretrained word embedding'''
        embeddings_index = {}
        f = open(embedding_path, 'r', encoding='utf-8')
        for line in f:
            values = line.split()
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print("Error on ", values[:2])
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index