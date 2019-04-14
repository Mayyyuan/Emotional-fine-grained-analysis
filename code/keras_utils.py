#coding=utf-8
import os
import codecs
import numpy as np
import gensim
import random
import json
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class TrieTree(object):
    def __init__(self):
        self.tree = {}

    def add(self, word):
        tree = self.tree
        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                tree[char] = {}
                tree = tree[char]
        tree['exist'] = True

    def search(self, word):
        tree = self.tree
        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                return False
        if 'exist' in tree and tree['exist'] == True:
            return True
        else:
            return False

#######################process the raw data file#############################
def get_data(data_file,label_file,max_len,n_tags,vocab,is_validation=False):
    # maskid = 0    ## padding class
    f = codecs.open(data_file,'r', 'utf8')
    f2 = codecs.open(label_file, 'r', 'utf8')
    docs = f.readlines()   ## list
    index_docs = word2index(docs, vocab, max_len)  ## list, have padded to certain length

    if (is_validation):
        return np.array(docs), index_docs,
    else:
        labels = f2.readlines()
        labelRes = []
        for label in labels:
            label = list(label.strip())
            label = [int(tag) for tag in label]
            labelRes.append(label)
        labels = pad_sequences(maxlen=max_len, sequences=labelRes, padding='post', truncating='post', value=0)
        labels = [to_categorical(i, num_classes=n_tags) for i in labels]  # one-hot編碼
        ## list [[00111222222],[1110000002222]]
        return np.array(docs), index_docs, labels

def get_test_data(test_file,vocab_path,train_max_len):
    f = codecs.open(test_file, 'r', 'utf8')
    docs = f.readlines()   ## list
    vocab_dir = os.path.join(vocab_path, 'vocab2')
    with open(vocab_dir, "r") as f2:
        vocab = json.load(f2)
    index_docs = word2index(docs, vocab, train_max_len)  ## list
    # res = np.array(index_docs)
    f.close()
    return index_docs, np.array(docs)

def gen_vocab(docs, maskid=0):
    word_set = set(word for doc in docs for word in doc.strip())  # ['中' ‘美’ ‘国’]
    vocab = {word: index + 1 for index, word in enumerate(word_set)}
    vocab['mask'] = maskid  # add 'mask' padding word
    vocab[u'unk'] = len(vocab)  # add 'unk'
    return vocab

def word2index(docs, vocab, train_max_len=0):
    index_docs = []
    for doc in docs:
        index_list = []
        for word in doc:
                if word not in vocab:
                        word = u'unk'
                index_list.append(vocab[word])
        index_docs.append(index_list)
    # if(train_max_len==0):
    #     train_max_len = max([len(doc) for doc in index_docs])
    # index_docs = [doc + [vocab['mask']] * (train_max_len - len(doc)) for doc in index_docs]
    index_docs = pad_sequences(maxlen=train_max_len, sequences=index_docs, padding='post', truncating='post', value=0)
    return index_docs   ## list [[2380,  512, 1254, ...,    0,    0,    0],[1548,...]]

#########################w2v_model#######################################
def get_word_vector(vocab, w2v_model, embedding_size):
    word_vector = []
    unkown_vector = [i/(embedding_size+0.0) for i in range(embedding_size)]
    word_vector.append(unkown_vector)
    wv_model = gensim.models.Word2Vec.load(w2v_model)
    #sorted by value
    vocab_ordered = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
    for word in vocab_ordered.keys():
        if word in wv_model:
                word_vector.append(wv_model[word])
        else:
                word_vector.append([random.random() for i in range(embedding_size)])
    # word_vector.pop()
    return np.array(word_vector)

#######################################生成一个batch_size的迭代器##########################################
def batch_iter(data_size, data, batch_size, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(list(data))
    #data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    # print(num_batches_per_epoch)
    # for epoch in range(num_epochs):
        # Shuffle the data at each epoch
    if shuffle:
        # shuffle_indices = np.random.permutation(np.arange(data_size))
        # shuffled_data = data[shuffle_indices]
        shuffle_indices = random.sample(range(0, data_size), data_size)  ### shuffle
        data_x, data_y, data_doc = zip(*data)
        shuffled_data_x = [data_x[i] for i in shuffle_indices]
        shuffled_data_y = [data_y[i] for i in shuffle_indices]
        shuffled_data_doc = [data_doc[i] for i in shuffle_indices]
    else:
        shuffled_data_x, shuffled_data_y, shuffled_data_doc = zip(*data)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)   # 解决最后一个batch溢出的问题
        # yield (shuffled_data[start_index:end_index], start_index)
        yield zip(shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index],shuffled_data_doc[start_index:end_index])

def add_label(string,dictionary_list,max_length):
    label_all = []
    for dictionary in dictionary_list:
        right_pointer = len(string)
        flag = []
        # B:0,E:1,S:2,O:3,I:4
        # 01234，BIESO
        left_pointer = 0
        while (True):
            if left_pointer == len(string):
                break
            while (True):
                segmention = string[left_pointer:right_pointer]
                segmention_len = len(segmention)
                if segmention:  # if not null
                    #	if segmention in dictionary:
                    if dictionary.search(segmention):
                        if segmention_len == 1:
                            flag.append(3)  ## single
                        if segmention_len == 2:
                            flag.append(0)  ## begin
                            flag.append(2)  ## end
                        if segmention_len > 2:
                            flag.append(0)  ## begin
                            for k in range(1, segmention_len - 1):
                                flag.append(1)  ## inside
                            flag.append(2)  ## end
                        left_pointer = right_pointer
                        break
                    else:
                        right_pointer -= 1
                        continue
                else:
                    flag.append(4)  ## outside
                    left_pointer = right_pointer + 1
                    break
            right_pointer = len(string)
        label_all.append(flag)
    label_all = np.array(label_all)  ## [2, seq_length(max_length)], because our dic list have 2 files
    label_mat = []
    for index in range(max_length):
        if index <= len(string) - 1:
            label_mat.append(label_all[:, index])  ## index position's different tag
        else:
            label_mat.append(np.array(len(dictionary_list) * [4]))
    label_mat = np.array(label_mat)   ## [seq_length, 2]
    index_for_embedding = []
    for word in label_mat:
        index = 0
        for i in range(len(word)):  ## i=0 or i=1
            index = index + pow(5, i) * int(word[i])
        index_for_embedding.append(index)   ## range is (0,5^2-1=24)
    return np.array(index_for_embedding)

def get_dict(file_list):
    dictionary_list = []
    # dict_file_list=["test_地区","test_指标","test_主体"]
    dict_file_list = file_list
    for dict_file in dict_file_list:
        print(dict_file)
        tree = TrieTree()
        with codecs.open( dict_file, "r", "utf-8") as f:
            for line in f:
                line = line.strip().split('\t')
                area = line[0]
                tree.add(area)
        dictionary_list.append(tree)
    return dictionary_list

def matrix_index(sentence_matrix,dictionary_list,max_length):
    label_matrix = []
    for sentence in sentence_matrix:
        index = add_label(sentence, dictionary_list, max_length)
        label_matrix.append(index)
    return np.array(label_matrix)

def sentiment2vec(sentiment,wv_model,Flag):
    sen_vec = [2.18] * 100
    if Flag == 'word':
        #根据字模型生成情感词向量
        word_num = len(sentiment)
        for word in sentiment:
            if(word in wv_model):
                sen_vec = [wv_model[word][i]+sen_vec[i] for i in range(len(sen_vec))]
        sen_vec = [sen/word_num for sen in sen_vec]
    else:
        #根据词模型生成情感词向量
        if (sentiment.strip() in wv_model):
            sen_vec = [wv_model[sentiment][i] + sen_vec[i] for i in range(len(sen_vec))]
    return sen_vec

def evaluateVal(model, valid_x, valid_label_file, valid_docs, valid_dict_index, wv_for_score_model, score_clf,window=5):
    predictArr = model.predict([valid_x, valid_dict_index])
    predict = np.argmax(predictArr, axis=-1)
    flag = 'word'
    tp = 0
    fp = 0
    fn1 = 0
    fn2 = 0
    with open(valid_label_file,'r',encoding='utf8') as f:
        for i, line in enumerate(f.readlines()):
            sentence = valid_docs[i]
            sentence_len = len(sentence)
            cur_predict = predict[i][:sentence_len]
            predict_theme = []
            predict_senti = []
            j = 0
            while(j<len(cur_predict)):
                cur_label = cur_predict[j]
                if(cur_label==0):
                    j+=1
                    continue
                elif(cur_label==4):
                    predict_senti.append([sentence[j],j,j])  ## word, start_index, end_index
                    j+=1
                elif(cur_label==8):
                    predict_theme.append([sentence[j],j,j])
                    j+=1
                elif(cur_label==1):
                    senti_start = j
                    j+=1
                    while(j<len(cur_predict)):
                        if(cur_predict[j]==3):
                            predict_senti.append([sentence[senti_start:j+1],senti_start,j])
                            j+=1
                            break
                        elif(cur_predict[j]==2):
                            j+=1
                        else:
                            break
                elif (cur_label == 5):
                    theme_start = j
                    j += 1
                    while (j<len(cur_predict)):
                        if (cur_predict[j] == 7):
                            predict_theme.append([sentence[theme_start:j + 1],theme_start,j])
                            j+=1
                            break
                        elif(cur_predict[j] == 6):
                            j += 1
                        else:
                            break
                else:
                    j+=1
#######################################################rule match method################################################
            result = []
            for senti_tuple in predict_senti:
                senti_word = senti_tuple[0]
                sen_vec = sentiment2vec(senti_word, wv_for_score_model, Flag=flag)
                sen_vec = np.array([sen_vec])
                score = int(score_clf.predict(sen_vec)[0])
                senti_start = senti_tuple[1]
                senti_end = senti_tuple[2]
                first = True
                is_match = False
                for theme_tuple in predict_theme:
                    if (abs(senti_start - theme_tuple[2]) < window or abs(senti_end - theme_tuple[1]) < window):
                        if (not first and theme_tuple[1] > senti_start):  ## 保证不被情感词后面的主题词更新
                            result.pop()
                        result.append([theme_tuple[0], senti_word, score])
                        is_match = True
                        first = False
                if (not is_match):
                    result.append(['NULL', senti_word, score])
#############################################################calc F1 score##############################################
            ## result format is like: [['NULL', '满意', 1], ['NULL', '实惠', 1], ['NULL', '方便', 1]]
            line = line.strip().split(',')
            if(len(line[0].strip())==0):
                fn1+=len(result)
            elif(len(result)==0):
                fn2+=len(line[0][:-1].split(';'))
            else:
                theme = line[0][:-1].split(';')
                sentiment = line[1][:-1].split(';')
                senti_score = line[2][:-1].split(';')
                a = len(theme)
                b = len(result)
                if(a<b):
                    fn1+=(b-a)
                elif(a>b):
                    fn2+=(a-b)
                for i in range(b):
                    right_flag = False
                    for j in range(a):
                        if(sentiment[j]!=result[i][1] or int(senti_score[j])!=result[i][2]):
                            continue
                        else:
                            # if(theme[j] in result[i][0] or result[i][0] in theme[j]):
                            theme_list = [item for item in theme[j]]
                            pred_theme_list = [item for item in result[i][0]]
                            intersection = [v for v in theme_list if v in pred_theme_list]
                            if(len(intersection)!=0):
                                tp+=1
                                right_flag=True
                                break
                    if(not right_flag):
                        fp+=1
    precision = tp/(tp+fp)
    recall = tp/(tp+fp+fn2)
    f1 = 2*(precision*recall)/(precision+recall)
    return f1
