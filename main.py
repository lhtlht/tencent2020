import pandas as pd
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings("ignore")

path_buld = "../data/tencent2020/build/"
path_save = "../data/tencent2020/save/"
train_preliminary_p = path_buld + "train_preliminary/"

if __name__ == '__main__':
    offline = True
    user = pd.read_csv(train_preliminary_p + "user.csv", encoding='utf-8')
    train = pd.read_csv(path_save + "train.csv", encoding='utf-8')
    test = pd.read_csv(path_save + "test.csv", encoding='utf-8')
    if offline:
        user_train, user_test = train_test_split(user, test_size=0.2, random_state=2020)
        user_train = user_train.merge(train, how='left', on='user_id')
        user_test = user_test.merge(train, how='left', on='user_id')
    else:
        user_train, user_test = user, test
        user_train = user_train.merge(train, how='left', on='user_id')





    print(user_train.head())


    count_vector = CountVectorizer()

    vecot_matrix = count_vector.fit_transform(np.concatenate((user_train['creative_id'],user_test['creative_id']), axis=0))

    print(vecot_matrix.shape)


    ## TF-IDF
    train_tfidf = TfidfTransformer(use_idf=True).fit_transform(vecot_matrix)

    clf = MultinomialNB().fit(train_tfidf, user_train['gender'])
