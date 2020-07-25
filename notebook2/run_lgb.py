import sys
import os
import logging
sys.path.append(r"..")
from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
import gc
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
path_build = "./build/"
path_embed = "./embed/"
path_list = "./feature_series/"
path_mnt = "./mnt/"
path_sub = "./sub/"
users = pd.read_pickle(f"{path_build}/train_user.pkl")
users.sort_values(by=['user_id'], ascending=[True], inplace=True)
users = users.reset_index(drop=True)
def tfidf_matrix_p(features_list, min_dfs):
    for fea, min_df in zip(features_list, min_dfs):
        tfidfer = TfidfVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b", min_df=1, max_features=min_df)  # (?u)\b\w+\b"，这样就不会忽略单个的字符
        fea_list = pd.Series(np.load(f"{path_list}{fea}_list.npy", allow_pickle=True)).map(lambda x:' '.join(x)).values
        print(fea)
        if fea == 'creative_id':
            tfidf_matrix = tfidfer.fit_transform(fea_list)
            #save_pickle(tfidfer.vocabulary_, pickle_path+f'creative_id_tfidf_v_{min_df}')
            #save_pickle(tfidf_matrix, pickle_path+f'creative_id_tfidf_m_{min_df}')

            print(f'tfidf matrix {fea} size:{tfidf_matrix.shape}')
        else:
            tfidf_matrix_fea = tfidfer.fit_transform(fea_list)
            print(f'tfidf matrix {fea} size:{tfidf_matrix_fea.shape}')
            tfidf_matrix = csr_matrix(sparse.hstack((tfidf_matrix, tfidf_matrix_fea)))
            #save_pickle(tfidfer.vocabulary_, pickle_path+f'{fea}_tfidf_v_{min_df}')
            #save_pickle(tfidf_matrix_fea, pickle_path+f'{fea}_tfidf_m_{min_df}')
    
    #save_pickle(tfidf_matrix, pickle_path+f'tfidf_m')
   
    print(f'tfidf matrix size:{tfidf_matrix.shape}')
    return tfidf_matrix

tfidf_features = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'click_times',
                 'creative_id_times', 'ad_id_times',  'product_id_times', 'advertiser_id_times',
                 'product_category_times', 'industry_times',]
tfidf_features = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'click_times',
                 ]
max_features = [900000, 900000, 300000, 300000, 300000, 300000, 300000]
tfidf_matrix = tfidf_matrix_p(tfidf_features, max_features)


    
params = {'num_leaves': 63,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': 2,
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "num_threads": 35,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu' 
          }

def lr_pred(pred_y, n_classes, offline, cv):
    params['num_class'] = n_classes
    if offline:
        train_x = tfidf_matrix[0:2700000]
        train_y = users[pred_y][0:2700000] - 1
    else:
        train_x = tfidf_matrix[0:3000000]
        train_y = users[pred_y][0:3000000] - 1

    val_x = tfidf_matrix[2700000:3000000]
    val_y = users[pred_y][2700000:3000000] - 1
    
    test_x =  tfidf_matrix[3000000:4000000]
    seed = 2020
    if cv:
        n_fold = 5
        count_fold = 0
        preds_list = list()
        oof = np.zeros((train_x.shape[0], n_classes))
        kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        kfold = kfolder.split(train_x, train_y)
        for train_index, vali_index in kfold:
            print("training......fold",count_fold)
            count_fold = count_fold + 1
            k_x_train = train_x[train_index]
            k_y_train = train_y[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_y[vali_index]
            
            trn_data = lgb.Dataset(k_x_train, label=k_y_train)
            val_data = lgb.Dataset(k_x_vali, label=k_y_vali)
            clf = lgb.train(params,
                    trn_data,
                    1000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    early_stopping_rounds=200)
            oof[vali_index] = clf.predict(k_x_vali, num_iteration=clf.best_iteration)
            pred = clf.predict(test_x, num_iteration=clf.best_iteration)
            preds_list.append(pred)
            print(pred)
        return preds_list, oof
    else:
        reg = LogisticRegression(solver='saga', multi_class='ovr', n_jobs=-1)
        reg_model = reg.fit(train_x, train_y)
        val_proba = reg_model.predict_proba(val_x)
        
        val_pre = np.argmax(val_proba,axis=1) + 1
        accuracy = accuracy_score(val_y, val_pre)
        print(f'{pred_y}-accuracy:{accuracy}')
        return 1,1

offline = False
cv = True    
gender_prob, gender_oof = lr_pred('gender',2, offline, cv)
age_prob, age_oof = lr_pred('age',10, offline, cv)
if offline == False:
    gender_oof_pre = np.argmax(gender_oof,axis=1) + 1
    age_oof_pre = np.argmax(age_oof,axis=1) + 1
    gender_accuracy = accuracy_score(users['gender'][:3000000], gender_oof_pre)
    age_accuracy = accuracy_score(users['age'][:3000000], age_oof_pre)

    print(gender_accuracy,age_accuracy,gender_accuracy+age_accuracy)

    gender_prob_mean = np.array(gender_prob).mean(axis=0)
    gender_pre = np.argmax(gender_prob_mean,axis=1) + 1

    age_prob_mean = np.array(age_prob).mean(axis=0)
    age_pre = np.argmax(age_prob_mean,axis=1) + 1

    np.save(f"{path_sub}/val_gender_prob.npy", gender_oof)
    np.save(f"{path_sub}/val_age_prob.npy", age_oof)
    np.save(f"{path_sub}/gender_prob.npy", gender_prob_mean)
    np.save(f"{path_sub}/age_prob.npy", age_prob_mean)
