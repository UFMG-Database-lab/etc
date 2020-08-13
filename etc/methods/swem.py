from base import Tokenizer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np
from os import path

from tensorflow.python import pywrap_tensorflow

import os
GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
from keras.utils import to_categorical

# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from math import floor

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    qtd_batch = max(1, int(np.ceil(n/minibatch_size)))
    for i in range(qtd_batch):
        minibatch_end = minibatch_start + minibatch_size
        batch = idx_list[minibatch_start:minibatch_end]
        while len(batch) < minibatch_size:
            batch = np.concatenate( (batch, idx_list[:(minibatch_size-len(batch))]), axis=0 )
        minibatches.append(batch)
        minibatch_start += minibatch_size

    return minibatches

class SWEM(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, train_val_proportion=.1, pacience=20,
                    fix_emb = False, reuse_w = True, reuse_cnn = False,
                    reuse_discrimination = True, restore = True, tanh = True,
                    model = 'cnn_deconv', permutation = 0, substitution = 's',
                    W_emb = None, cnn_W = None, cnn_b = None, maxlen = 305,
                    filter_shape = 5, embed_size = 300, lr = 3e-4, layer = 3,
                    stride = [2, 2], batch_size = 128, max_epochs = 1000,
                    n_gan = 500, L = 100, drop_rate = 0.8, encoder = 'concat',
                    part_data = False, portion = 0.001, print_freq = 500,
                    valid_freq = 500, discrimination = False, dropout = 0.5,
                    H_dis = 300, device='/cpu:0', pretreined_embeddings=None):
        super().__init__( "SWEM", use_validation=True )

        self.tokenizer = Tokenizer()
        self.device = device
        self.pacience = pacience

        self.fix_emb = fix_emb
        self.reuse_w = reuse_w
        self.reuse_cnn = reuse_cnn # reuse cnn for discrimination
        self.reuse_discrimination = reuse_discrimination
        self.restore = restore
        self.tanh = tanh # activation fun for the top layer of cnn, otherwise relu
        self.model = model # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = permutation
        self.substitution = substitution # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.H_dis = H_dis

        self.W_emb = W_emb
        self.cnn_W = cnn_W
        self.cnn_b = cnn_b
        self.maxlen = maxlen
        self.filter_shape = filter_shape
        self.embed_size = embed_size
        self.lr = lr
        self.layer = layer
        self.stride = stride # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_gan = n_gan # self.filter_size * 3
        self.L = L
        self.drop_rate = drop_rate
        self.encoder = encoder  # 'max' 'concat'

        self.part_data = part_data
        self.portion = portion # 10%  1%  float(sys.argv[1])

        self.pretreined_embeddings = pretreined_embeddings
        self.train_val_proportion = train_val_proportion

        self.print_freq = print_freq
        self.valid_freq = valid_freq

        self.discrimination = discrimination
        self.dropout = dropout

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train
        y_train = y_train
        X_val = X_val
        y_val = y_val

        tf.reset_default_graph()
        # Convert text to array of termid
        self.n_class = len(set(y_train))
        X_train = list(map(self.tokenizer.doc2termid, X_train))
        X_val   = list(map(self.tokenizer.transform, X_val))

        y_train = self.to_categorical(y_train)
        y_val   = self.transform_to_categorical(y_val)
        
        # ixtoword = [ term ] // the term position is the termid
        # wordtoix = { term: termid }

        self.n_words = len(self.tokenizer.term2ix)
        self.ixtoword = self.tokenizer.get_ix2term()

        if self.part_data:
            np.random.seed(42)
            train_ind = np.random.choice(len(X_train), int(len(X_train)*self.portion), replace=False)
            X_train = [X_train[t] for t in train_ind]
            y_train = [y_train[t] for t in train_ind]

        self.fix_emb = False
        if self.pretreined_embeddings is not None and path.exists(self.pretreined_embeddings):
            params = np.load(self.pretreined_embeddings)
            if params['Wemb'].shape == (self.n_words, self.embed_size):
                self.W_emb = params['Wemb']
            else:
                self.fix_emb = False

        with tf.device(self.device):
            self.x_ = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, self.maxlen])
            self.x_mask_ = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, self.maxlen])
            self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, self.n_class])
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            accuracy_, loss_, train_op, W_emb_, self.predictions_, self.H_enc_ = self.emb_classifier(self.x_, self.x_mask_, self.y_, self.keep_prob)

        max_val_accuracy = 0.

        config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        #saver = tf.compat.v1.train.Saver()

        print("###############################################################")
        best_acc = -1.
        last_improvement = 0

        sess = tf.compat.v1.Session(config=config)

        #with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(self.max_epochs):
            print("Starting epoch %d" % epoch)
            uidx = 0
            kf = get_minibatches_idx(len(X_train), self.batch_size, shuffle=True)
            for train_index in kf:
                print(uidx, end='\r')
                uidx += 1
                sents = [X_train[t] for t in train_index]
                x_labels = [y_train[t] for t in train_index]
                x_labels = np.array(x_labels)
                x_labels = x_labels.reshape((len(x_labels), self.n_class))

                x_batch, x_batch_mask = self.prepare_data_for_emb(sents)
                _, loss = sess.run([train_op, loss_], feed_dict={self.x_: x_batch, self.x_mask_: x_batch_mask, self.y_: x_labels, self.keep_prob: self.drop_rate})
            print()
            val_correct = 0.0
            kf_val = get_minibatches_idx(len(X_val), self.batch_size, shuffle=True)
            size_last = len(X_val) % self.batch_size
            for val_index in kf_val:
                val_sents  = [X_val[t] for t in val_index]
                val_labels = np.array([y_val[t] for t in val_index])
                val_labels = val_labels.reshape((len(val_labels), self.n_class))
                x_val_batch, x_val_batch_mask = self.prepare_data_for_emb(val_sents)

                val_accuracy = sess.run(accuracy_, feed_dict={self.x_: x_val_batch, self.x_mask_: x_val_batch_mask, self.y_: val_labels, self.keep_prob: 1.0})
                val_correct += val_accuracy * len(val_index)

            val_accuracy = val_correct / len(X_val)
            last_improvement +=1
            if val_accuracy > best_acc:
                # Save the best model
                self.sess = sess
                best_acc = val_accuracy
                last_improvement = 0
            elif last_improvement > self.pacience:
                break
            print(f"Validation accuracy {val_accuracy:.5}/{best_acc:.5}({last_improvement}/{self.pacience})")
   
    def transform(self, X):
        doc_termid = list(map(self.tokenizer.transform, X))
        kf_transf = get_minibatches_idx(len(X), self.batch_size, shuffle=False)
        result = []
        for i,transf_index in enumerate(kf_transf):
            transf_sents  = [doc_termid[t] for t in transf_index]
            x_transf_batch, x_transf_batch_mask = self.prepare_data_for_emb(transf_sents)
            H_enc = self.sess.run(self.H_enc_,  feed_dict={self.x_: x_transf_batch, self.x_mask_: x_transf_batch_mask, self.keep_prob: 1.0})
            result.append( H_enc )

        bla = np.concatenate(result, axis=0)
        bla = bla[:len(X)]

        return bla
              
    def predict(self, X):
        doc_termid = list(map(self.tokenizer.transform, X))
        kf_transf = get_minibatches_idx(len(X), self.batch_size, shuffle=False)
        result = []
        for i,transf_index in enumerate(kf_transf):
            transf_sents  = [doc_termid[t] for t in transf_index]
            x_transf_batch, x_transf_batch_mask = self.prepare_data_for_emb(transf_sents)
            predictions = self.sess.run(self.predictions_,  feed_dict={self.x_: x_transf_batch, self.x_mask_: x_transf_batch_mask, self.keep_prob: 1.0})
            result.append(predictions)

        bla = np.concatenate(result, axis=0)
        bla = bla[:len(X)]

        return self.from_categorical(bla)
    
    def __del__(self):
        tf.reset_default_graph()
    
    def to_categorical(self,y):
        self.ix2label = list(set(y))
        self.label2ix = {}
        for i,v in enumerate(self.ix2label):
            code = np.zeros(self.n_class)
            code[i] = 1.
            self.label2ix[v] = code
        return self.transform_to_categorical(y)
    def transform_to_categorical(self, y):
        return np.array([ self.label2ix[v] for v in y ])
    def from_categorical(self,y):
        return np.array([ self.ix2label[v] for v in y ])
    
    def embedding(self, features, prefix='', is_reuse=None):
        """Customized function to transform batched x into embeddings."""
        # Convert indexes of words into embeddings.
        #  b = tf.compat.v1.get_variable('b', [self.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
        with tf.compat.v1.variable_scope(prefix + 'embed', reuse=is_reuse):
            if self.fix_emb:  #####################    #####################################################################################
                ###################################    #####################################################################################
                ################  use pre-trained word-embedding as W  #####################################################################
                ################  ajust  #####################################################################
                ###################################    #####################################################################################
                ###################################    #####################################################################################
                assert (hasattr(self, 'W_emb'))
                assert (np.shape(np.array(self.W_emb)) == (self.n_words, self.embed_size))
                W = tf.compat.v1.get_variable('W', initializer=self.W_emb, trainable=True)
                #pdb.set_trace()
                print("initialize word embedding finished")
            else:
                weightInit = tf.random_uniform_initializer(-0.001, 0.001)
                W = tf.compat.v1.get_variable('W', [self.n_words, self.embed_size], initializer=weightInit)
                # tf.stop_gradient(W)
        if hasattr(self, 'relu_w') and self.relu_w:
            W = tf.nn.relu(W)

        # W_norm = normalizing(W, 1)
        word_vectors = tf.nn.embedding_lookup(W, features)

        return word_vectors, W
    def discriminator_2layer(self, H, dropout, prefix='', num_outputs=1, is_reuse=None):
        # last layer must be linear
        # H = tf.squeeze(H, [1,2])
        # pdb.set_trace()
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=self.H_dis,
                                    biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                    reuse=is_reuse)
        logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                            biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
        return logits   
    def prepare_data_for_emb(self, seqs_x):
        maxlen = self.maxlen
        lengths_x = [len(s) for s in seqs_x]
        if maxlen != None:
            new_seqs_x = []
            new_lengths_x = []
            for l_x, s_x in zip(lengths_x, seqs_x):
                if l_x < maxlen:
                    new_seqs_x.append(s_x)
                    new_lengths_x.append(l_x)
                else:
                    new_seqs_x.append(s_x[:maxlen])
                    new_lengths_x.append(maxlen)
            lengths_x = new_lengths_x
            seqs_x = new_seqs_x

            if len(lengths_x) < 1:
                return None, None

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        #x = np.zeros((maxlen_x, n_samples)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x[idx, :lengths_x[idx]] = s_x
            # x_mask[idx, :lengths_x[idx]] = 1.
            x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token

        return x, x_mask
        
    def emb_classifier(self, x, x_mask, y, dropout):
        # print x.get_shape()  # batch L
        x_emb, W_emb = self.embedding(x)  # batch L emb
        x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
        x_emb = tf.nn.dropout(x_emb, dropout)   # batch L emb 1

        x_mask = tf.expand_dims(x_mask, axis=-1)
        x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

        x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
        H_enc = tf.reduce_sum(x_sum, axis=1, keepdims=True)  # batch 1 emb 1
        H_enc = tf.squeeze(H_enc)  # batch emb
        x_mask_sum = tf.reduce_sum(x_mask, axis=1, keepdims=True)  # batch 1 1 1
        x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1
        H_enc_1 = H_enc / x_mask_sum  # batch emb

        H_enc_2 = tf.nn.max_pool2d(x_emb, [1, self.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
        H_enc_2 = tf.squeeze(H_enc_2)

        H_enc = tf.concat([H_enc_1, H_enc_2], 1)

        H_enc = tf.squeeze(H_enc)
        logits = self.discriminator_2layer(H_enc, dropout, prefix='classify_', num_outputs=self.n_class, is_reuse=None)  # batch * 10
        prob = tf.nn.softmax(logits)

        predictions = tf.argmax(prob, 1)

        correct_prediction = tf.equal(predictions, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

        train_op = layers.optimize_loss(
            loss,
            tf.compat.v1.train.get_global_step(),
            optimizer='Adam',
            learning_rate=self.lr)

        return accuracy, loss, train_op, W_emb, predictions, H_enc