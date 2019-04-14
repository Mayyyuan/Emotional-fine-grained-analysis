'''
author:yanbiao
'''
from keras_utils import *
from kerasModel import *
import tensorflow as tf
import os
from keras.models import load_model
import json
from sklearn.externals import joblib
from keras_contrib.utils import save_load_utils

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.flags.DEFINE_integer('max_len', 85, 'the sentence max length')
tf.flags.DEFINE_integer('embedding_dim', 100, 'the word vector dimension')
tf.flags.DEFINE_integer('vocab_num', 100, 'the vocab number')
tf.flags.DEFINE_integer('dictFeatureNum', 25, 'the dict feature number')
tf.flags.DEFINE_integer('dict_embedding_size',64, 'the dict feature embedding size')
tf.flags.DEFINE_integer('bilstm_hidden_dim',64,'the vocab number')
tf.flags.DEFINE_integer('lstm_hidden_dim',64,'the vocab number')
tf.flags.DEFINE_integer('hidden_size', 50, 'the single direction RNN output dimension')
tf.flags.DEFINE_integer('cate_num', 9, 'the class number')
tf.flags.DEFINE_integer('window_size', 5, 'min distance between theme and sentiment')
tf.flags.DEFINE_float('learning_rate', 0.01, 'the learning rate')
tf.flags.DEFINE_float('train_rate', 0.9, 'the percentage of all data')
tf.flags.DEFINE_integer('batch_size', 128, 'the number of sentence fed to the network')
tf.flags.DEFINE_integer('epoch', 200, 'the number of using the all data')
tf.flags.DEFINE_float('dropout_rate', 0.3, 'keep prob of dropout layer')
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_epoch", 1, "Evaluate model on dev set after this many steps (default: 100)")

tf.flags.DEFINE_string('data_file', '../new_data/train_corpus.txt', 'the raw train data not contain the label')
tf.flags.DEFINE_string('label_file', '../new_data/train_label.txt', 'the label for train data')
tf.flags.DEFINE_string('val_data_file', '../new_data/val_corpus.txt', 'the raw train data not contain the label')
tf.flags.DEFINE_string('val_label_file', '../new_data/val_label.txt', 'the label for train data')
tf.flags.DEFINE_string('vocab_dir', '../new_data/vocab', 'the path to save the vocab dict json file')

tf.flags.DEFINE_string('model_path', '../result/model_1022.h5', 'the raw train data not contain the label')
tf.flags.DEFINE_string('val_acc_predict', '../result/acc_validation_only_bilstm.txt', 'validation accuracy')
tf.flags.DEFINE_string('train_loss', '../result/train_loss_1022.txt', 'training data loss')
tf.flags.DEFINE_string('result_file', '../result/predict_1027.txt', 'the predict result path')

tf.flags.DEFINE_string('w2v_model', '../model/gameAndyuliaokudata.model', 'the word angle w2v model')
tf.flags.DEFINE_string('score_model', '../model/train_model1126_2.m', 'predict sentiment.')
tf.flags.DEFINE_string('word2vec_model', '../model/model_word_100.model', 'w2v model for sentiment')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("is_train", True, 'this is a flag')

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()  ## vc use
# FLAGS._parse_flags()

def prepareData():
    docs = []
    with open(FLAGS.data_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            docs.append(line.strip())
    with open(FLAGS.val_data_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            docs.append(line.strip())
    vocab = gen_vocab(docs, maskid=0)
    train_docs, train_x, train_y = get_data(FLAGS.data_file, FLAGS.label_file, FLAGS.max_len, FLAGS.cate_num, vocab)
    val_docs, val_x = get_data(FLAGS.val_data_file, FLAGS.val_label_file, FLAGS.max_len, FLAGS.cate_num, vocab,is_validation=True)
    word_vector = get_word_vector(vocab, FLAGS.w2v_model, FLAGS.embedding_dim)
    print("word_vector size is "+str(word_vector.shape))
    jsObj = json.dumps(vocab)
    if not os.path.exists(FLAGS.vocab_dir):
        os.makedirs(FLAGS.vocab_dir)
    vocab_dir = os.path.join(FLAGS.vocab_dir, 'vocabulary')
    fileObject = open(vocab_dir, 'w', encoding='utf8')
    fileObject.write(jsObj)
    fileObject.close()

    FLAGS.vocab_num = len(vocab)
    sentence_num = len(train_x)
    sentence_length = train_x[0].shape[0]
    print("sentencce number is "+str(sentence_num)+", sentence length is "+str(sentence_length))

    print("train_x array shape:" + str(train_x.shape))
    print("train_y list shape:" +'('+ str(len(train_y))+','+str(train_y[0].shape[0])+','+str(train_y[0].shape[1])+')')
    print(train_x[0])
    print(train_y[0])
    print(train_docs[0])
    return train_x, np.array(train_y),train_docs, np.array(val_x),val_docs, word_vector

def train(train_x, train_y,train_docs,valid_x,valid_docs,word_vector,wv_for_score_model, score_clf):

    resultWriter = open(FLAGS.val_acc_predict,'w',encoding='utf8')
    trainloss = open(FLAGS.train_loss, 'w', encoding='utf8')

    model_object = dnnModel(FLAGS)
    model = model_object.build_bilstm_model(word_vector)
    maxF1 = 0.0
    file_list = ['../dictionary/trainSenDict', '../dictionary/theme1117.txt']
    dictionary_list = get_dict(file_list)
    valid_dict_index = matrix_index(valid_docs, dictionary_list, FLAGS.max_len)

    for i in range(FLAGS.epoch):
        resultWriter.write('recycle number is '+str(i)+'\n')
        trainloss.write('recycle number is '+str(i)+'\n')
        batchIter = batch_iter(len(train_x), zip(train_x, train_y,train_docs), FLAGS.batch_size, False)
        j = 0
        for zip_xydoc in batchIter:
            print(str(i)+' th epoch, '+str(j)+' th batch.')
            j += 1
            print('train.....')
            batch_x, batch_y, batch_doc = zip(*zip_xydoc)
            batch_x = np.array(list(batch_x))
            batch_y = np.array(list(batch_y))
            dict_index = matrix_index(batch_doc, dictionary_list, FLAGS.max_len)
            trainHistory = model.train_on_batch([batch_x, dict_index], batch_y)
            print("train loss is: " + str(trainHistory[0])+'\n')
            trainloss.write(str(trainHistory[0]))
            if(i>FLAGS.evaluate_epoch and j%FLAGS.checkpoint_every==0):
                f1 = evaluateVal(model, valid_x, FLAGS.val_label_file, valid_docs, valid_dict_index, wv_for_score_model, score_clf,window=FLAGS.window_size)
                print("the validation f1 score is " + str(f1))
                if(f1>maxF1):
                    print('####################update model#####################################################')
                    maxF1 = f1
                    resultWriter.write("the validation f1 is "+str(f1)+'\n')
                    predict(model, FLAGS.val_data_file, FLAGS.result_file, wv_for_score_model, score_clf,FLAGS.window_size,False)
                    resultWriter.flush()
                    trainloss.flush()
                    save_load_utils.save_all_weights(model, FLAGS.model_path)
            del batch_x
            del batch_y
        del batchIter
    trainloss.close()
    resultWriter.close()

def predict(model_path, test_data_path, result_file,wv_for_score_model, score_clf,window,word_vector,is_model_path=False):
    test_x, test_docs = get_test_data(test_data_path, FLAGS.vocab_dir, FLAGS.max_len)
    file_list = ['../dictionary/trainSenDict', '../dictionary/theme1117.txt']
    dictionary_list = get_dict(file_list)
    dict_index = matrix_index(test_docs, dictionary_list, FLAGS.max_len)
    print("test data is ok!")
    if(is_model_path):
        model_object = dnnModel(FLAGS)
        model = model_object.build_bilstm_model(word_vector)
        # model = model_object.build_crf_model(word_vector)
        save_load_utils.load_all_weights(model, model_path)
        # model = load_model(model)
    else:
        model = model_path
    print("model have loaded!")
    predictArr = model.predict([test_x, dict_index])
    predict = np.argmax(predictArr, axis=-1)
    flag = 'word'
    o = open(result_file,'w',encoding='utf8')

    for i in range(len(test_docs)):
        sentence = test_docs[i]
        sentence_len = len(sentence)
        cur_predict = predict[i][:sentence_len]
        predict_theme = []
        predict_senti = []
        j = 0
        while (j < len(cur_predict)):
            cur_label = cur_predict[j]
            if (cur_label == 0):
                j += 1
                continue
            elif (cur_label == 4):
                predict_senti.append([sentence[j], j, j])  ## word, start_index, end_index
                j += 1
            elif (cur_label == 8):
                predict_theme.append([sentence[j], j, j])
                j += 1
            elif (cur_label == 1):
                senti_start = j
                j += 1
                while (j < len(cur_predict)):
                    if (cur_predict[j] == 3):
                        predict_senti.append([sentence[senti_start:j + 1], senti_start, j])
                        j += 1
                        break
                    elif (cur_predict[j] == 2):
                        j += 1
                    else:
                        break
            elif (cur_label == 5):
                theme_start = j
                j += 1
                while (j < len(cur_predict)):
                    if (cur_predict[j] == 7):
                        predict_theme.append([sentence[theme_start:j + 1], theme_start, j])
                        j += 1
                        break
                    elif (cur_predict[j] == 6):
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
            score = str(score_clf.predict(sen_vec)[0])
            senti_start = senti_tuple[1]
            senti_end = senti_tuple[2]
            first = True
            is_match = False
            for theme_tuple in predict_theme:
                if (abs(senti_start - theme_tuple[2]) < window or abs(senti_end - theme_tuple[1]) < window):
                    # if (min(abs(senti_start - theme_tuple[2]), abs(senti_end - theme_tuple[1])) < min_distance):
                    if (not first and theme_tuple[1]>senti_start):
                        result.pop()
                    result.append([theme_tuple[0], senti_word, score])
                    is_match = True
                    first = False
            if (not is_match):
                result.append(['NULL',senti_word,score])
        cur_predict = [str(item) for item in cur_predict]
        o.write(str(result)[1:-1]+'\t'+''.join(cur_predict)+'\n')
    o.close()

if __name__ == "__main__":
    wv_for_score_model = gensim.models.Word2Vec.load(FLAGS.word2vec_model)
    score_clf = joblib.load(FLAGS.score_model)
    train_x, train_y,train_docs,valid_x,valid_docs, word_vector = prepareData()
    train(train_x, train_y, train_docs,valid_x, valid_docs, word_vector, wv_for_score_model, score_clf)
    # predict(FLAGS.model_path, FLAGS.val_data_file, FLAGS.result_file, wv_for_score_model, score_clf, FLAGS.window_size,word_vector, True)