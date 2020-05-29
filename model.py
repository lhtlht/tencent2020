import pandas as pd
import numpy as np
import sys
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy import sparse
from scipy.sparse import csr_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt; plt.style.use('seaborn')
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 1000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': -1,


}

def multi_column_LabelEncoder(df,columns,rename=True):
    le = LabelEncoder()
    for column in columns:
        #print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column])
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    print('LabelEncoder Successfully!')
    return df


def reg_model(train, test, label_name, model_type, numerical_features, category_features, seed, cv=True):
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    train.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True,drop=True)
    if model_type == 'rf':
        train.fillna(0, inplace=True)

    # combine = pd.concat([train, test], axis=0)
    # combine = multi_column_LabelEncoder(combine, category_features, rename=True)
    # combine[category_features] = combine[category_features].astype('category')
    # train = combine[:train.shape[0]]
    # test = combine[train.shape[0]:]

    features = category_features + numerical_features
    train_x = train[features]
    train_y = train[label_name]
    test_x = test[features]
    if cv:
        n_fold = 2
        count_fold = 0
        preds_list = list()
        oof = np.zeros(train_x.shape[0])
        kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        kfold = kfolder.split(train_x, train_y)
        for train_index, vali_index in kfold:
            print("training......fold",count_fold)
            count_fold = count_fold + 1
            k_x_train = train_x.loc[train_index]
            k_y_train = train_y.loc[train_index]
            k_x_vali = train_x.loc[vali_index]
            k_y_vali = train_y.loc[vali_index]
            if model_type == 'lgb':
                lgb_model = lgb.LGBMRegressor(**lgb_params)
                if 'sample_weight' in train.columns:
                    lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                              early_stopping_rounds=200, verbose=False, eval_metric="mae",
                                              sample_weight=train.loc[train_index]['sample_weight'],
                                              categorical_feature=category_features
                                              )
                else:
                    lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)],
                                          early_stopping_rounds=200, verbose=False, eval_metric="mae",
                                              categorical_feature=category_features)
                k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
                pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
            elif model_type == 'xgb':
                xgb_model = XGBRegressor(**xgb_params)
                xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                          early_stopping_rounds=200, verbose=False)
                k_pred = xgb_model.predict(k_x_vali)
                pred = xgb_model.predict(test_x)
            elif model_type == 'rf':
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, criterion="mae",n_jobs=-1,random_state=2019)
                model = rf_model.fit(k_x_train, k_y_train)
                k_pred = rf_model.predict(k_x_vali)
                pred = rf_model.predict(test_x)
            elif model_type == 'cat':
                ctb_params = {
                    'n_estimators': 1000,
                    'learning_rate': 0.02,
                    'random_seed': 4590,
                    'reg_lambda': 0.08,
                    'subsample': 0.7,
                    'bootstrap_type': 'Bernoulli',
                    'boosting_type': 'Plain',
                    'one_hot_max_size': 100,
                    'rsm': 0.5,
                    'leaf_estimation_iterations': 5,
                    'use_best_model': True,
                    'max_depth': 5,
                    'verbose': -1,
                    'thread_count': 4,
                    'cat_features':category_features
                }

                cat_model = cat.CatBoostRegressor(**ctb_params)
                cat_model.fit(k_x_train, k_y_train, verbose=False, use_best_model=True, eval_set=[(k_x_vali, k_y_vali)])
                k_pred = cat_model.predict(k_x_vali)
                pred = cat_model.predict(test_x)
            preds_list.append(pred)
            oof[vali_index] = k_pred

        # if model_type == 'lgb':
        #     feature_importance_df = pd.DataFrame({
        #         'column': features,
        #         'importance': lgb_model.feature_importances_,
        #     }).sort_values(by='importance')
        #     feature_importance_df.to_csv('feature_importance.csv', index=False, )
            #print(feature_importance_df)

            # plt.figure(figsize=(15, 5))
            # plt.barh(range(len(features)), lgb_model.feature_importances_)
            # plt.bar(range(len(features)), lgb_model.feature_importances_)
            # plt.xticks(range(len(features)), features, rotation=-45, fontsize=14)
            # plt.title('Feature importance', fontsize=14)
            # plt.show()
            # import shap
            # explainer = shap.TreeExplainer(lgb_model)
            # shap_values = explainer.shap_values(train_x)
            # player_explainer = pd.DataFrame()
            # player_explainer['feature'] = features
            # player_explainer['feature_value'] = train_x.iloc[10].values
            # player_explainer['shap_value'] = shap_values[10]
            # print(player_explainer)
            # shap.initjs()
            # aa = shap.force_plot(explainer.expected_value, shap_values[10], train_x.iloc[10])
            # #bb = shap.summary_plot(shap_values, train_x)
            # cc = shap.summary_plot(shap_values, train_x, plot_type="bar")

            #shap.save_html('aa.html', bb)
        preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        preds = list(preds_df.mean(axis=1))

        return preds, oof
    else:
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model = lgb_model.fit(train_x, train_y,eval_metric='mse')
        preds = lgb_model.predict(test_x)
        oof = lgb_model.predict(train_x)
        return preds, oof



def class_model(train, test, features_map, model_type='lgb', class_num=2, cv=True):
    label = features_map['label']
    label_features = features_map['label_features']
    category_onehot_features = features_map['category_features1']
    category_features = features_map['category_features2']
    numerical_features = features_map['numerical_features']
    combine = pd.concat([train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, label_features, rename=True)
    combine.reset_index(inplace=True)

    onehoter = OneHotEncoder()
    X_onehot = onehoter.fit_transform(combine[category_onehot_features])
    train_x_onehot = X_onehot.tocsr()[:train.shape[0]].tocsr()
    test_x_onehot = X_onehot.tocsr()[train.shape[0]:].tocsr()
    train_x_original = combine[numerical_features+category_features][:train.shape[0]]
    test_x_original = combine[numerical_features+category_features][train.shape[0]:]
    train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()
    train_y = combine[label][:train.shape[0]]

    train_x2 = combine.loc[:train.shape[0]-1]
    test_x2 = combine.loc[train.shape[0]:]
    train_x2[label] = train_x2[label].astype(np.int)
    features = category_features + numerical_features + category_onehot_features
    train_y2 = train_x2[label]
    train_x2 = train_x2[features]
    test_x2 = test_x2[features]



    #模型训练
    lgb_params = {
        'application': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 31,
        'verbosity': -1,
        'data_random_seed': 2019,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 5,
        'device':'cpu'
    }
    cat_model = cat.CatBoostClassifier(iterations=1000, depth=8, cat_features=features, learning_rate=0.05, custom_metric='F1',
                               eval_metric='F1', random_seed=2019,
                               l2_leaf_reg=5.0, logging_level='Silent')
    # clf = lgb.LGBMClassifier(
    #     objective='binary',
    #     learning_rate=0.02,
    #     n_estimators=1000,
    #     max_depth=-1,
    #     num_leaves=31,
    #     subsample=0.8,
    #     subsample_freq=1,
    #     colsample_bytree=0.8,
    #     random_state=2019,
    #     reg_alpha=1,
    #     reg_lambda=5,
    #     n_jobs=6
    # )
    cxgb = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        random_state=2019,
        n_jobs=6
    )
    if cv:
        n_fold = 5
        print(train.shape[0])
        result = np.zeros((test.shape[0],))
        oof = np.zeros((train.shape[0],))
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)
        kfold = skf.split(train_x, train_y)
        count_fold = 0
        for train_index, vali_index in kfold:
            print("training......fold",count_fold)
            count_fold = count_fold + 1
            k_x_train = train_x[train_index]
            k_y_train = train_y.loc[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_y.loc[vali_index]

            if model_type == 'lgb':
                trn = lgb.Dataset(k_x_train, k_y_train)
                val = lgb.Dataset(k_x_vali, k_y_vali)
                lgb_model = lgb.train(lgb_params, train_set=trn, valid_sets=[trn, val],
                              num_boost_round=5000,early_stopping_rounds=200, verbose_eval=-1)
                test_pred_proba = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
                val_pred_proba = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration)
                # clf.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                # test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
                # val_pred_proba = clf.predict_proba(k_x_vali, num_iteration=clf.best_iteration_)
            elif model_type == 'xgb':
                cxgb.fit(k_x_train, k_y_train,eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],early_stopping_rounds=200, verbose=False)
                test_pred_proba = cxgb.predict_proba(test_x)
                val_pred_proba = cxgb.predict_proba(k_x_vali)
            elif model_type == 'cat':
                cat_model.fit(k_x_train,k_y_train)
                test_pred_proba = cat_model.predict(test_x)
                val_pred_proba = cat_model.predict(k_x_vali)
            result = result + test_pred_proba
            oof[vali_index] = val_pred_proba
        result = result/n_fold
    else:

        print(train_x.shape, train_y.shape)
        if model_type == 'cat':
            cat_model.fit(train[features], train[label])
            test_pred_proba = cat_model.predict(test[features])
            train_pred_proba = cat_model.predict(train[features])
        else:
            lgb_df = lgb.Dataset(train_x2, train_y2)
            lgb_model = lgb.train(lgb_params, train_set=lgb_df, categorical_feature=category_features,
                                  num_boost_round=1500,)

            test_pred_proba = lgb_model.predict(test_x2)
            train_pred_proba = lgb_model.predict(train_x2)
            feat_imp = lgb_model.feature_importance(importance_type='gain')
            feat_nam = lgb_model.feature_name()
            for fn, fi in zip(feat_nam, feat_imp):
                print(fn,fi)
        # clf.fit(train_x, train_y, categorical_feature=category_features)
        # #test_pred = clf.predict(test_x)
        # test_pred_proba = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)
        # train_pred_proba = clf.predict_proba(train_x, num_iteration=clf.best_iteration_)


        result = test_pred_proba
        oof = train_pred_proba

    return oof,result


class LSTModel2(keras.Model):
    def __init__(self, units, num_classes, voc_size, emb_size, emb_mat, max_len):
        super(LSTModel2, self).__init__()
        self.units = units
        self.embedding = keras.layers.Embedding(voc_size, emb_size, input_length=max_len, trainable=False, weights=[emb_mat])
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(self.units))
        self.dense1 = keras.layers.Dense(self.units, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None, mask=None):
        # x1 = self.embedding(x['a'])
        # x2 = self.embedding(x['b'])
        # e = tf.concat([x1,x2],1)
        # x = self.lstm(e)
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
