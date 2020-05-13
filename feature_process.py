from utils import *
import pandas as pd
import numpy as np

path_buld = "../data/tencent2020/build/"
path_save = "../data/tencent2020/save/"


train_click_log = pd.read_csv(path_save + "train_click_log.csv", encoding="utf-8")
test_click_log = pd.read_csv(path_save + "test_click_log.csv", encoding="utf-8")



