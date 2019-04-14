import tensorflow as tf


class AnnotationModel(object):
    def __init__(self, FLAGS):
        self.x = tf.placeholder(tf.int32, [None, FLAGS.time_step], name='input')
        ## 这个地方不能使用batch_size，因为train的时候可能就不是batch的长度，predict的时候，是直接输入的整个test作为batch
        self.y = tf.placeholder(tf.int32, [None, FLAGS.time_step], name='label')
        self.w2v_result = tf.placeholder(tf.float32, [FLAGS.vocab_num, FLAGS.embedding_size], name='w2v_result')
        self.dict_vector = tf.placeholder(tf.int32, [None, FLAGS.time_step], name='dictSeries')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        mask_x = tf.sign(self.x)   ## 对于句子来说,maskid刚好是0,sign有两个值，0和1
        self.sequence_length = tf.reduce_sum(mask_x, axis=1)  # the length of each row in input_x
        ## 在模型里面通过这种机制来记录batch中每一个句子的实际长度，作为RNN模型的参数，刚好是这种能够排除maskid的机制

        with tf.name_scope('embedding'):
            emb_W = tf.Variable(tf.truncated_normal([FLAGS.vocab_num, FLAGS.embedding_size], 0., 1.), name='embedding')
            tf.assign(emb_W, self.w2v_result)
            em_X = tf.nn.embedding_lookup(emb_W, self.x)
            emb_W2 = tf.Variable(tf.truncated_normal([FLAGS.dictFeatureNum, FLAGS.dict_embedding_size], 0., 1.),
                                 name='embeddingDict')
            em_X2 = tf.nn.embedding_lookup(emb_W2, self.dict_vector)
            input = tf.concat([em_X, em_X2], 2, name='concatEmbedding')
            outputs = self.biRNN(input, FLAGS)
        with tf.name_scope('properties_softmax_loss'):
            outputs = tf.reshape(outputs, [-1, FLAGS.hidden_size * 2])
            soft_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_size * 2, FLAGS.class_num], 0., 1.),
                                 name='for_softmaxW')
            soft_b = tf.Variable(tf.truncated_normal([FLAGS.class_num]), name='for_softmaxB')
            outputs = tf.nn.xw_plus_b(outputs, soft_w, soft_b)
            softmax_probility = tf.nn.softmax(outputs, name='softmax_probility')
            softmax_result = tf.argmax(softmax_probility, axis=1, name='softmax_result')
            self.softmax_result2 = tf.reshape(softmax_result, [-1, FLAGS.time_step, 1], name='softmax_result2')
            test_predict = tf.reshape(self.softmax_result2, [-1, FLAGS.time_step])
            test_predict = tf.cast(test_predict, dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(test_predict, self.y), dtype=tf.float32), name='accuracy')
            y = tf.reshape(self.y, [-1])
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
            self.loss = tf.reduce_mean(self.loss)
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)        # AdamOptimizer
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def biRNN(self, input_batch, FLAGS):
        rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
        rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.keep_prob)
        rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
        rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.keep_prob)
        # tf.nn.bidirectional_dynamic_rnn表示调用n次call函数
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw,
                                                          dtype=tf.float32, sequence_length=self.sequence_length,
                                                          inputs=input_batch, time_major=False)
        outputs = tf.concat(outputs, 2)
        return outputs
