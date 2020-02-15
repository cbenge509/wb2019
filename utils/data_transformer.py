import numpy as np
import re
import string

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utils.data_helper import data_loader

class DataTransformer(object):

    def __init__(self, max_num_words, max_sequence_length, testsize=0.1):
        self.data_loader = data_loader.DataLoader()
        self.clean_word_dict = self.data_loader.load_clean_words("C:/edx/competition/data/cleanwords.txt")
        self.train_df = self.data_loader.load_dataset("C:/edx/competition/data/train_values.csv", "row_id")
        self.test_df = self.data_loader.load_dataset("C:/edx/competition/data/test_values.csv", "row_id")
        self.labels = self.data_loader.load_dataset("C:/eds/competition/data/train_labels.csv", "row_id")

        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.list_classes = list(self.labels.columns)

        self.train_df, self.val_df, self.labels, self.val_labels = train_test_split(self.train_df, self.labels, 
                                                random_state=42, test_size=testsize, shuffle=True)

    def prepare_data(self):
        list_sentences_train = self.train_df["doc_text"].values
        list_sentences_val = self.val_df["doc_text"].values
        list_sentences_test = self.test_df["doc_text"].values

        print("Preprocessing data...")

        self.train_docs = [self.clean_text(text) for text in list_sentences_train]
        self.val_docs = [self.clean_text(text) for text in list_sentences_val]
        self.test_docs = [self.clean_text(text) for text in list_sentences_test]

        self.build_tokenizer(self.train_docs + self.test_docs + self.val_docs)
        train_sequences = self.tokenizer.texts_to_sequences(self.train_docs)
        val_sequences = self.tokenizer.texts_to_sequences(self.val_docs)
        test_sequences = self.tokenizer.texts_to_sequences(self.test_docs)

        training_labels = self.labels[self.list_classes].values
        validation_labels = self.val_labels[self.list_classes].values

        print("Preprocessing complete..")

        return train_sequences, training_labels, val_sequences, validation_labels, test_sequences

    def clean_text(self, text, clean_custom_stopwords=True):

        text = text.lower()
        text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
        text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)
    
        if clean_custom_stopwords:
            # Drop dataset stopwords
            text = re.sub(r"\.jpg", " ", text)
            text = re.sub(r"\.png", " ", text)
            text = re.sub(r"\.gif", " ", text)
            text = re.sub(r"\.bmp", " ", text)
            text = re.sub(r"\.pdf", " ", text)
            text = re.sub(r"image:", " ", text)
            text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
            text = re.sub(r"\{\|[^\}]*\|\}", " ", text) # clean the css part
            text = re.sub(r"\[?\[user:.*\]", " ", text)
            text = re.sub(r"\[?\[user:.*\|", " ", text)
            text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
            text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
            text = re.sub(r"\[?\[special:.*\]", " ", text)
            text = re.sub(r"\[?\[special:.*\|", " ", text)
            text = re.sub(r"\[?\[category:.*\]", " ", text)
            text = re.sub(r"\[?\[category:.*\|", " ", text)

            text = re.sub(r"wp:", " ", text)
            text = re.sub(r"file:", " ", text)

        for typo, correct in self.clean_word_dict.items():
            text = re.sub(typo, " " + correct + " ", text)
            
        text = re.sub(r"´", "'", text)
        text = re.sub(r"—", " ", text)
        text = re.sub(r"’", "'", text)
        text = re.sub(r"‘", "'", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"i’m", "i am", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"–", " ", text)
        text = re.sub(r"−", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"#", " # ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"<3", " love ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"world bank group", "worldbankgroup ", text)
        text = re.sub(r"world bank", "worldbankgroup ", text)
        text = re.sub(r"worldbankgroup org", "worldbankgroup ", text)
        text = re.sub(r"\s{2,}", " ", text)

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub(' ', text)  #remove punctuation (replace with space for now)

        self.word_count_dict = defaultdict(int)
        text = text.split()
        for t in text:
            self.word_count_dict[t] += 1
        text = " ".join(text)

        return (text)

    def build_embedding_matrix(self, embeddings_index):
        nb_words = min(self.max_num_words, len(embeddings_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('null-word.txt', 'w', encoding='utf-8')

        for word, i in word_index.items():

            if i >= self.max_num_words:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix

    def build_tokenizer(self, comments):
        self.tokenizer = Tokenizer(num_words=self.max_num_words, char_level=False)
        self.tokenizer.fit_on_texts(comments)
        self.tokenizer.word_index = {e:i for e,i in self.tokenizer.word_index.items() if i <= self.max_num_words}
