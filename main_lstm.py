import sys
import os
import logging
sys.path.append(r"..")
from utils import *
from model import *
import numpy as np
import pandas as pd
from gensim import models

from collections import defaultdict
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.client import device_lib

print(tf.__version__)
print(tf.test.is_built_with_gpu_support)
print(tf.test.is_gpu_available())
print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
gpus = tf.config.experimental.list_physical_devices('GPU')

path_build = "../data/tencent2020/build/"
path_save = "../data/tencent2020/save/"
pickle_path = "../data/tencent2020/pickle/"
sub_path = "../data/tencent2020/sub/"
model_path = "../data/tencent2020/model/"
train_preliminary_p = path_build + "train_preliminary/"


#读取数据
user_ids = pd.read_pickle(f"{pickle_path}/user_ids_relencode.pickle")
user = pd.read_csv(train_preliminary_p + "user.csv", encoding='utf-8')
user_ids = user_ids.merge(user, how='left', on='user_id')
del user
print('user_ids shape',user_ids.shape)

# 超参数
vocab_size = 3500000
max_length = 200
embedding_dim = 100
units = 128
num_classes = 2
batch_size = 256
epochs = 2

# embedding matrix
creative_id_v = models.KeyedVectors.load_word2vec_format(f"{path_save}/creative_id_w2v_128.bin", binary=True)
ad_id_v = models.KeyedVectors.load_word2vec_format(f"{path_save}/ad_id_w2v_128.bin", binary=True)
advertiser_id_v = models.KeyedVectors.load_word2vec_format(f"{path_save}/advertiser_id_w2v_64.bin", binary=True)
product_id_v = models.KeyedVectors.load_word2vec_format(f"{path_save}/product_id_w2v_64.bin", binary=True)

creative_id_em = np.zeros((vocab_size, creative_id_v.vector_size))
ad_id_em = np.zeros((vocab_size, ad_id_v.vector_size))
for w in creative_id_v.vocab:
    creative_id_em[int(w)] = creative_id_v[w]
for w in ad_id_v.vocab:
    ad_id_em[int(w)] = ad_id_v[w]


creative_id_train_seq = keras.preprocessing.sequence.pad_sequences(user_ids['creative_id'][:10000],value = 0,padding = 'post',maxlen = max_length )
creative_id_val_seq = keras.preprocessing.sequence.pad_sequences(user_ids['creative_id'][810000:900000],value = 0,padding = 'post',maxlen = max_length )
creative_id_test_seq = keras.preprocessing.sequence.pad_sequences(user_ids['creative_id'][900000:],value = 0,padding = 'post',maxlen = max_length )

ad_id_train_seq = keras.preprocessing.sequence.pad_sequences(user_ids['ad_id'][:10000],value = 0,padding = 'post',maxlen = max_length )
ad_id_val_seq = keras.preprocessing.sequence.pad_sequences(user_ids['ad_id'][810000:900000],value = 0,padding = 'post',maxlen = max_length )
ad_id_test_seq = keras.preprocessing.sequence.pad_sequences(user_ids['ad_id'][900000:],value = 0,padding = 'post',maxlen = max_length )

gender_train_label = np.array(user_ids['gender'][:10000])
gender_val_label = np.array(user_ids['gender'][810000:900000])

def input_fn(feature_dict, label=None, epochs=5, shuffle=True, batch_size=64, fit_key='train'):
    if fit_key == 'train':
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict, label))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict))
    if shuffle:
        dataset = dataset.shuffle(100*batch_size)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset



class LSTModel(keras.Model):
    def __init__(self, units, num_classes, voc_size, emb_size, max_len):
        super(LSTModel, self).__init__()
        self.units = units
        self.embedding1 = keras.layers.Embedding(voc_size, emb_size, input_length=max_len, trainable=False, weights=[creative_id_em])
        self.embedding2 = keras.layers.Embedding(voc_size, emb_size, input_length=max_len, trainable=False, weights=[ad_id_em])

        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(self.units))
        self.dense1 = keras.layers.Dense(self.units, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, feature_dict, training=None, mask=None):
        x1 = self.embedding1(feature_dict['creative_id'])
        x2 = self.embedding2(feature_dict['ad_id'])
        e = tf.concat([x1, x2], -1)
        x = self.lstm(e)
        # x = self.embedding(x)
        # x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


model = LSTModel(units=units,
                 num_classes=num_classes,
                 voc_size=vocab_size,
                 emb_size=embedding_dim,
                 max_len=max_length)
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
train_feature_dict = {'creative_id': creative_id_train_seq,
                'ad_id': ad_id_train_seq}
val_feature_dict = {'creative_id': creative_id_val_seq,
                    'ad_id': ad_id_val_seq}

train_dataset = input_fn(train_feature_dict, gender_train_label-1, epochs=2, shuffle=False, batch_size=512)
val_dataset = input_fn(val_feature_dict, gender_val_label-1, epochs=1, shuffle=False, batch_size=64)
model.fit(train_dataset,gender_train_label-1, validation_data=(val_dataset, gender_val_label-1))
#model.save(model_path+'gender', save_format='tf')
model.summary()

# config = tf.estimator.RunConfig(
#     log_step_count_steps=1,
# )

tf.keras.estimator.model_to_estimator(model)
estimator = keras.estimator.model_to_estimator(model)
#estimator.train(input_fn=lambda: input_fn(train_feature_dict, gender_train_label-1, epochs=2, shuffle=True, batch_size=512))

