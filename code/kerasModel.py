from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Input, TimeDistributed,Concatenate
from keras import optimizers
from keras_contrib.layers.crf import CRF

class dnnModel():
    def __init__(self, args):
        self.vocab_num = args.vocab_num
        self.cate_num = args.cate_num
        self.maxlen = args.max_len
        self.bilstm_hidden_dim = args.bilstm_hidden_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.embedding_dim = args.embedding_dim
        self.dropout_rate = args.dropout_rate
        self.lr = args.learning_rate
        self.dictFeatureNum = args.dictFeatureNum
        self.dict_embedding_size = args.dict_embedding_size
    def build_crf_model(self,word_vector):
        print("load model")
        left = Input(shape=(self.maxlen,))
        left_embedding = Embedding(self.vocab_num+1, self.embedding_dim, weights=[word_vector], input_length=self.maxlen, mask_zero=True, trainable=True)(left)

        right = Input(shape=(self.maxlen,))
        right_embedding = Embedding(self.dictFeatureNum, self.dict_embedding_size, input_length=self.maxlen,mask_zero=True, trainable=True)(right) ## 0~24

        model = Concatenate(axis=-1)([left_embedding, right_embedding])

        model = Bidirectional(LSTM(self.bilstm_hidden_dim, recurrent_dropout=self.dropout_rate, return_sequences=True))(model)
        model = LSTM(self.lstm_hidden_dim, recurrent_dropout=self.dropout_rate, return_sequences=True)(model)
        model = TimeDistributed(Dense(50, activation="relu"))(model)
        # model = TimeDistributed(Dense(50, activation="relu"))(model)
        crf = CRF(self.cate_num)
        out = crf(model)
        adam = optimizers.Adam(lr=self.lr)
        # output = Dense(self.catenum, activation='softmax')(single_drop)
        model = Model(input=[left, right], output=out)
        model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def build_bilstm_model(self, word_vector):
        print("load model")
        left = Input(shape=(self.maxlen,))
        left_embedding = Embedding(self.vocab_num + 1, self.embedding_dim, weights=[word_vector],input_length=self.maxlen, mask_zero=True, trainable=True)(left)

        right = Input(shape=(self.maxlen,))
        right_embedding = Embedding(self.dictFeatureNum, self.dict_embedding_size, input_length=self.maxlen,mask_zero=True, trainable=True)(right)  ## 0~24

        model = Concatenate(axis=-1)([left_embedding, right_embedding])
        model = Bidirectional(LSTM(self.bilstm_hidden_dim, recurrent_dropout=self.dropout_rate, return_sequences=True))(model)
        out = TimeDistributed(Dense(self.cate_num, activation="softmax"))(model)
        model = Model(input=[left, right], output=out)
        adam = optimizers.Adam(lr=self.lr)
        model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def build_bi_singlelstm_model(self, word_vector):
        print("load model")
        left = Input(shape=(self.maxlen,))
        left_embedding = Embedding(self.vocab_num + 1, self.embedding_dim, weights=[word_vector],input_length=self.maxlen, mask_zero=True, trainable=True)(left)

        right = Input(shape=(self.maxlen,))
        right_embedding = Embedding(self.dictFeatureNum, self.dict_embedding_size, input_length=self.maxlen,mask_zero=True, trainable=True)(right)  ## 0~24

        model = Concatenate(axis=-1)([left_embedding, right_embedding])
        model = Bidirectional(LSTM(self.bilstm_hidden_dim, recurrent_dropout=0.1, return_sequences=True))(model)
        model = LSTM(self.lstm_hidden_dim, recurrent_dropout=self.dropout_rate, return_sequences=True)(model)
        out = TimeDistributed(Dense(self.cate_num, activation="softmax"))(model)
        model = Model(input=[left, right], output=out)
        adam = optimizers.Adam(lr=self.lr)
        model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
        return model

