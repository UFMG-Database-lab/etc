from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import tensorflow as tf
import numpy as np

class LSTMClassifier(BaseEstimator, ClassifierMixin, TransformerMixin, tf.keras.Model):
    """
        Defines a Long Short Term Memory (LSTM) based model as an approach in the text classification problem.
    """

    def __init__(self, vocabulary_size=50000, sequence_size=256, hidden_size=300,
                learning_rate=0.001, patience=10, epochs=100, batch_size=16,
                dropout=.2, dropout_lstm=.2, dropout_rec=.2, device='/cpu:0'):

        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.dropout = dropout
        self.dropout_lstm = dropout_lstm
        self.dropout_rec = dropout_rec
        self.device = device

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(self.vocabulary_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the RNNTextClassifier model for a maximum number of epochs.
        :param x_train: input tensor with shape (m:samples, n: max text length).
        :param y_train: target tensor with shape (m:samples, n: number of classes).
        :param x_val: validation tensor with shape (v:samples, n: max text length).
        :param y_val: target tensor with shape (v:samples, n: number of classes).
        :return: the training history.
        """ 
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.transform_text(X_train)
        X_val   = self.transform_text(X_val)
        
        y_train = self.to_categorical(y_train)
        y_val   = self.to_categorical(y_val, train=False)

        # model definition
        with tf.device(self.device):
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocabulary_size, self.hidden_size),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.LSTM(self.hidden_size, dropout=self.dropout_lstm, recurrent_dropout=self.dropout_rec),
                tf.keras.layers.Dense(self.n_class, activation='softmax'),
            ])
            print(dir(self.model))
            # configures the model for training
            self.model.compile(loss='categorical_crossentropy',
                            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                            metrics=['accuracy'])

            self.model.fit(x=X_train,
                            y=y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=self.get_callbacks()
                            )
        return self

    def predict(self, X_test):
        """
        :param x: input tensor with shape (m:samples, n: max text length).
        :return: predicted tensor with shape (m:samples, n: number of classes).
        """
        x = self.transform_text(X_test)
        y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, 1)
        y_pred = self.from_categorical(y_pred)
        return y_pred

    def predict_proba(self, X_test):
        """
        :param x: input tensor with shape (m:samples, n: max text length).
        :return: predicted tensor with shape (m:samples, n: number of classes).
        """
        x = self.transform_text(X_test)
        y_pred = self.model.predict(x)
        return y_pred

    def transform_text(self, x):
        x = self.tokenizer.texts_to_sequences(x)
        x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence_size)
        return x

    def to_categorical(self,y, train=True):
        if train:
            self.ix2label = list(set(y))
            self.n_class = len(self.ix2label)
            self.label2ix = {}
            for i,v in enumerate(self.ix2label):
                code = np.zeros(self.n_class)
                code[i] = 1.
                self.label2ix[v] = code
        return np.array([ self.label2ix[v] for v in y ])
        
    def from_categorical(self,y):
        return np.array([ self.ix2label[v] for v in y ])

    def get_callbacks(self):
        """
        :return: ModelCheckpoint and EarlyStopping callbacks.
        """
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./checkpoint",
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                save_frequency=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                min_delta=0.0001
            )
        ]