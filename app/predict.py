# ライブラリのインポート
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import joblib

# 前処理
train = pd.read_csv('./static/csv/train_x.csv')

train = train.dropna(thresh=10000, axis=1)

train['拠点番号'] = train['拠点番号'].astype('category').cat.codes

train_y = pd.read_csv('./static/csv/train_y.csv')
train = pd.concat([train,train_y], axis=1)

X = train.loc[:,['フラグオプション選択', '派遣形態', '紹介予定派遣', '正社員登用あり','給与/交通費　給与下限','学校・公的機関（官公庁）','経験者優遇','会社概要　業界コード','車通勤OK','拠点番号']].values
y = train.loc[:,['応募数 合計']].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# モデル選択
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

from sklearn.ensemble import RandomForestRegressor
param_grid = {'n_estimators': [450,500],
              'max_depth': [8,10]}
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(RandomForestRegressor(), param_grid, cv=kf)

# 学習
gs.fit(X_train,y_train)

# 学習済みモデルの保存
joblib.dump(gs, "predict.pkl", compress=True)