"""
CRIM：人均犯罪率

ZN：25,000平方英尺以上民用土地的比例

INDUS：城镇非零售业商用土地比例

CHAS：是否邻近查尔斯河，1是邻近，0是不邻近

NOX：一氧化氮浓度（千万分之一）

RM：住宅的平均房间数

AGE：自住且建于1940年前的房屋比例

DIS：到5个波士顿就业中心的加权距离

RAD：到高速公路的便捷度指数

TAX：每万元的房产税率

PTRATIO：城镇学生教师比例

B： 1000(Bk − 0.63)2 其中Bk是城镇中黑人比例

LSTAT：低收入人群比例

ans：自住房中位数价格，单位是千元
"""

import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
# Series: df["sepal length (cm)"]
df["ans"] = boston["target"]
df

# 進入sklean以後, 我們就不要用df, numpy array
import numpy as np
from sklearn.model_selection import train_test_split

y = np.array(df["ans"])
# axis參數
x = np.array(df.drop(["ans"], axis=1))
# 90% x, 10% x, 90% y, 10% y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# train_test_split([1, 2, 3, 4], ["a", "b", "c", "d"], test_size=0.25)
# numpy: .shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=5)
reg.fit(x_train, y_train)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(14, 14))
plot_tree(reg,
          feature_names=boston["feature_names"],
          filled=True,
          max_depth=2)

from sklearn.metrics import r2_score
pre = reg.predict(x_test)
r2_score(y_test, pre)