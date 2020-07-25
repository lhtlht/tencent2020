import sys
import os
import logging
sys.path.append(r"..")
import numpy as np
import pandas as pd
import deepctr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score
from gensim import models
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix

from collections import defaultdict
from collections import Counter


# if not os.path.exists(pickle_path):
#     os.mkdir(pickle_path)
# if not os.path.exists(sub_path):
#     os.mkdir(sub_path)

def multi_column_LabelEncoder(df,columns,rename=True):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in columns:
        print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column]) + 1
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    print('LabelEncoder Successfully!')
    return df

def multi_column_LabelEncoderOrder(df,columns,rename=True, max_features=None):
    for column in columns:
        print(column,"LabelEncoderOrder......")
        feaValueCounts = df[column].value_counts()
        df[column+"_index"] = df[column].map( dict(zip( feaValueCounts.index, range(1,len(feaValueCounts)+1)  )) )
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
        if max_features:
            df.loc[df[column]>max_features, column] = max_features
    print('LabelEncoder Successfully!')
    return df

def view_dfna(df):
    columns = df.columns
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': columns,
                                     'percent_missing': percent_missing})
    return missing_value_df

def search_weight(valid_y, raw_prob, init_weight=[1.0]*10,class_num=10, step=0.001):
    weight = init_weight.copy()
    f_best = accuracy_score(y_true=valid_y, y_pred=raw_prob.argmax(
        axis=1))
    flag_score = 0
    round_num = 1
    while(flag_score != f_best):
        print("round: ", round_num)
        round_num += 1
        flag_score = f_best
        for c in range(class_num):
            for n_w in range(0, 2000,10):
                num = n_w * step
                new_weight = weight.copy()
                new_weight[c] = num

                prob_df = raw_prob.copy()
                prob_df = prob_df * np.array(new_weight)

                f = accuracy_score(y_true=valid_y, y_pred=prob_df.argmax(
                    axis=1))
                if f > f_best:
                    weight = new_weight.copy()
                    f_best = f
                    print(f)
    return weight


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    #return df

def save_pickle(val, path):
    import pickle
    f = open(path, 'wb')
    pickle.dump(val, f)
    f.close()


def load_pickle(path):
    import pickle
    f = open(path, 'rb')
    val = pickle.load(f)
    f.close()
    return val


def get_array_label(label_prob, con=0):
    pre = label_prob.argmax(axis=1) + con
    return pre


