from utils import *
import pandas as pd
import numpy as np

path_buld = "../data/tencent2020/build/"
path_save = "../data/tencent2020/save/"
train_preliminary_p = path_buld + "train_preliminary/"
test_p = path_buld + "test/"

train_ad = pd.read_csv(train_preliminary_p + "ad.csv", encoding="utf-8")
train_click_log = pd.read_csv(train_preliminary_p + "click_log.csv", encoding="utf-8")
train_user = pd.read_csv(train_preliminary_p + "user.csv", encoding="utf-8")

train_ad['industry'] = train_ad['industry'].replace('\\N', '0').astype(int)
train_ad['product_id'] = train_ad['product_id'].replace('\\N', '0').astype(int)

train_click_log = train_click_log.merge(train_ad, how='left', on='creative_id')
#train_preliminary = train_preliminary.merge(train_user, how='left', on='user_id')


test_ad = pd.read_csv(test_p + "ad.csv", encoding="utf-8")
test_click_log = pd.read_csv(test_p + "click_log.csv", encoding="utf-8")
test_ad['industry'] = test_ad['industry'].replace('\\N', '0').astype(int)
test_ad['product_id'] = test_ad['product_id'].replace('\\N', '0').astype(int)

test_click_log = test_click_log.merge(test_ad, how='left', on='creative_id')

print(train_click_log.info())
print(test_click_log.info())

train_click_log = reduce_mem_usage(train_click_log)
test_click_log = reduce_mem_usage(test_click_log)

train_click_log.to_csv(path_save + "train_click_log.csv", encoding='utf-8', index=False)
test_click_log.to_csv(path_save + "test_click_log.csv", encoding='utf-8', index=False)

