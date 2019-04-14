'''
tensorboard --logdir="./graphs/l2" --port 6006      see the tensorboard
http://localhost:6006/
'''
from utils2 import *
from model import *
# from for_dict import *
import time
import os
import json
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(device_lib.list_local_devices())

tf.flags.DEFINE_integer('time_step',300,'the sentence max length')
tf.flags.DEFINE_integer('embedding_size', 100, 'the word vector dimension')
tf.flags.DEFINE_integer('dictFeatureNum', 5, 'the dict feature number')
tf.flags.DEFINE_integer('vocab_num', 100, 'the vocab number')
tf.flags.DEFINE_integer('dict_embedding_size', 64, 'the dict feature embedding size')
tf.flags.DEFINE_integer('hidden_size', 64, 'the single direction RNN output dimension')
tf.flags.DEFINE_integer('class_num', 9, 'the class number')
tf.flags.DEFINE_float('learning_rate', 0.01, 'the learning rate')
tf.flags.DEFINE_float('train_rate', 0.98, 'the percentage of all data')
tf.flags.DEFINE_string('w2v_model', '../data/gameAndyuliaokudata.model', 'the word angle w2v model')
tf.flags.DEFINE_integer('batch_size', 50, 'the number of sentence fed to the network')
tf.flags.DEFINE_integer('epoch', 200, 'the number of using the all data')
tf.flags.DEFINE_float('keep_prob', 0.7, 'keep prob of dropout layer')
tf.flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_string('data_file', '../data/train', 'the raw train data not contain the label')
tf.flags.DEFINE_string('label_file', '../data/train_label', 'the label for train data')
tf.flags.DEFINE_string('vocab_dir', '../data/vocab', 'the path to save the vocab dict json file')
tf.flags.DEFINE_string('trainloss_dir', '../data/trainloss_see_train_0709.txt', 'train loss record')
tf.flags.DEFINE_string('validationLoss_dir', '../data/validationLoss_see_train_0709.txt', 'train loss record')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("is_train", True, 'this is a flag')

FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

max_len = FLAGS.time_step

docs, index_docs, labels, vocab = get_data(FLAGS.data_file, FLAGS.label_file, max_len)
word_vector = get_word_vector(vocab, FLAGS.w2v_model, FLAGS.embedding_size)

jsObj = json.dumps(vocab)
if not os.path.exists(FLAGS.vocab_dir):
    os.makedirs(FLAGS.vocab_dir)
vocab_dir = os.path.join(FLAGS.vocab_dir, 'vocab')
fileObject = open(vocab_dir, 'w')
fileObject.write(jsObj)
fileObject.close()


FLAGS.vocab_num = len(vocab)

sentence_num, sentence_length = labels.shape
# FLAGS.time_step = sentence_length
FLAGS.time_step = 300

train_x = index_docs[: int(FLAGS.train_rate * sentence_num)]
train_y = labels[: int(FLAGS.train_rate * sentence_num)]
train_docs = docs[: int(FLAGS.train_rate * sentence_num)]

valid_x = index_docs[int(FLAGS.train_rate * sentence_num):]
valid_y = labels[int(FLAGS.train_rate * sentence_num):]
valid_docs = docs[int(FLAGS.train_rate * sentence_num):]

print("train_x shape:" + str(train_x.shape))
print("train_y shape:" + str(train_y.shape))
print("valid_x shape:" + str(valid_x.shape))
print("valid_y shape:" + str(valid_y.shape))
print(train_x[0])


###########################################训练模型##################################
def train_step(x, y, word_vector, dict_vector):
    feed_dict = {
        model.w2v_result: word_vector,
        model.dict_vector: dict_vector,
        model.x: x,
        model.y: y,
        model.keep_prob: FLAGS.keep_prob
    }
    loss, step, _, loss_summary = sess.run([model.loss, model.global_step, model.train_op, model.summary_op], feed_dict)
    return loss, loss_summary


def validation_step(x, y, word_vector, valid_dict_vector):
    feed_dict = {
        model.w2v_result: word_vector,
        model.dict_vector: valid_dict_vector,
        model.x: x,
        model.y: y,
        model.keep_prob: 1.0
    }
    acc = sess.run(model.acc, feed_dict)
    # test_predict = tf.reshape(test_predict, [-1, FLAGS.time_step])
    ## 下面的操作会被画在graph里面，也就是lazy_loading的问题
    # acc = tf.reduce_mean(tf.cast(tf.equal(test_predict, y), dtype=tf.float32))
    # accRes = sess.run(acc)
    return acc


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)  # 将graph载入到一个会话session中
    # sess = tf.Session()

    with sess.as_default():
        model = AnnotationModel(FLAGS)
        sess.run(tf.global_variables_initializer())

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))

        print('writing to {}\n'.format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        # checkpoint_prefix = os.path.abspath(os.path.join(checkpoint_dir, 'checkpoints'))
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # if not os.path.exists(checkpoint_prefix):
        #     os.makedirs(checkpoint_prefix)
########################################变量初始化，准备加载图模型###################################################

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../graphs/l2', sess.graph)
        ret = batch_iter(train_x.shape[0], zip(train_x, train_y, train_docs), FLAGS.batch_size, FLAGS.epoch, False)
        saver = tf.train.Saver()
        file_list = ['trainSenDict']
        dictionary_list = get_dict(file_list)
        valid_dict_vector = matrix_index(valid_docs, dictionary_list, FLAGS.time_step)
        validationAccLog = {}
        patience_cnt = 0
        trainLossPreStep = 0
        patience = 16
        o_trainloss = open(FLAGS.trainloss_dir, 'w', encoding='utf8')
        o_validation_loss = open(FLAGS.validationLoss_dir, 'w', encoding='utf8')
        # minValidationLoss = 10000.0
        minTrainLoss = 10000.0
        for batch in ret:
            x_, y_, doc_ = zip(*batch[0])     # zip(*) is equal to unzip
            dict_vector = matrix_index(doc_, dictionary_list, FLAGS.time_step)
            train_loss, loss_summary = train_step(x_, y_, word_vector, dict_vector)
            current_step = tf.train.global_step(sess, model.global_step)
            writer.add_summary(loss_summary, global_step=current_step)
            epoch_step = current_step * FLAGS.batch_size / 20000
            o_trainloss.write("epoch " + str(epoch_step) + '\t' + "train loss is:" + str(train_loss) + '\n')
            print('current_epoch is {}，current_step is {}， the train mean loss is {}'.format(epoch_step, current_step, train_loss))
#             if current_step % FLAGS.evaluate_every == 0:
#                 vali_acc = validation_step(valid_x, valid_y, word_vector, valid_dict_vector)
#                 print("\nEvaluation:")
#                 print('current_epoch is {}，current_step is {}， the train mean loss is {}'.format(epoch_step
#         , current_step, train_loss))
#                 print('\nValidation Acc is:')
#                 print(vali_acc)
#                 validationAccLog[current_step] = vali_acc
# ########################################early stopping###################################################
#                 min_delta = 0.001
#                 if epoch_step > 0 and trainLossPreStep - train_loss > min_delta:
#                     patience_cnt = 0
#                 else:
#                     patience_cnt += 1
#                 if patience_cnt > patience:
#                     print("early stopping...")
#                     break
#                 trainLossPreStep = train_loss
##########################################################################################################
            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_dir, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))
            if current_step % FLAGS.evaluate_every == 0:
                o_trainloss.flush()  # 刷新文件
                vali_acc = validation_step(valid_x, valid_y, word_vector, valid_dict_vector)
                o_validation_loss.write("epoch " + str(epoch_step) + '\t' + "validation loss is:" +
                                        str(vali_acc) + '\n')
                o_validation_loss.flush()
                # if (vali_acc < minValidationLoss):
                if (train_loss < minTrainLoss):
                    # minValidationLoss = vali_acc
                    minTrainLoss = train_loss
                    print("\nEvaluation:")
                    print('current_epoch is {}，current_step is {}， the train mean loss is {}'.format(epoch_step, current_step,train_loss))
                    print('\nValidation Acc is:{}'.format(vali_acc))
                    validationAccLog[current_step] = vali_acc
                    path = saver.save(sess, checkpoint_dir, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

    writer.close()
    o_trainloss.close()
    o_validation_loss.close()
    valiAccList = sorted(validationAccLog.items(), key=lambda d: d[1], reverse=True)
    accDir = os.path.abspath(os.path.join(checkpoint_dir, 'validationAccuracy'))
    with open(accDir, 'w') as o:
        for item in valiAccList:
            o.write(str(item[0])+'\t'+str(item[1])+'\n')
    print("Optimization Finished!")
