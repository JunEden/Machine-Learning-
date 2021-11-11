from urllib.request import urlretrieve
url = "https://github.com/Elwing-Chou/tibaml0922/raw/main/train.csv"
urlretrieve(url, "train.csv")
url = "https://github.com/Elwing-Chou/tibaml0922/raw/main/test.csv"
urlretrieve(url, "test.csv")

import pandas as pd
data = pd.read_csv("train.csv", encoding="utf-8")
predict = pd.read_csv("test.csv", encoding="utf-8")

predict

df = pd.concat([data, predict],
                axis=0,
                ignore_index=True)
df = df.drop(["PassengerId", "Survived"],
             axis=1)
df

def cabin(c):
    if not pd.isna(c):
        return c[0]
    else:
        return None
df["Cabin"] = df["Cabin"].apply(cabin)
df

dic = df["Ticket"].value_counts()
df["Ticket"] = df["Ticket"].replace(dic)

def name(n):
    n = n.split(",")[-1].split(".")[0]
    n = n.strip()
    return n
df["Name"] = df["Name"].apply(name)
# name("Kelly, Mr. James")

nasum = df.isna().sum()
# 篩選: Series[跟你資料筆數一樣多的T/F]
nasum[nasum > 0].sort_values(ascending=False)

med = df.median().drop(["Pclass"])
df = df.fillna(med)
nasum = df.isna().sum()
# 篩選: Series[跟你資料筆數一樣多的T/F]
nasum[nasum > 0].sort_values(ascending=False)

most = df["Embarked"].value_counts().idxmax()
df["Embarked"] = df["Embarked"].fillna(most)
nasum = df.isna().sum()
# 篩選: Series[跟你資料筆數一樣多的T/F]
nasum[nasum > 0].sort_values(ascending=False)

nc = df["Name"].value_counts()
whitelist = nc[nc > 50].index
def namemid(n):
    if n not in whitelist:
        return None
    else:
        return n
df["Name"] = df["Name"].apply(namemid)

df = pd.get_dummies(df)
df = pd.get_dummies(df, columns=["Pclass"])
df

df["Family"] = df["SibSp"] + df["Parch"]
df

# loc[列標籤], iloc[第幾列]
import numpy as np
x = df.iloc[:data.shape[0]]
x = np.array(x)
x_predict = df.iloc[data.shape[0]:]
x_predict = np.array(x_predict)
y = np.array(data["Survived"])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier()
params = {
    "max_depth":range(5, 10),
    "n_estimators":range(21, 100, 2)
}
search = GridSearchCV(clf, params, cv=10, n_jobs=-1)
search.fit(x, y)

print(search.best_score_)
print(search.best_params_)

from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(max_depth=7, 
                             n_estimators=51)
scores = cross_val_score(clf, x, y, cv=10, n_jobs=-1)
print(scores)
print(np.average(scores))

clf = RandomForestClassifier(max_depth=7, 
                             n_estimators=51)
clf.fit(x, y)
pre = clf.predict(x_predict)
result = pd.DataFrame({
    "PassengerId":predict["PassengerId"],
    "Survived":pre
})
result.to_csv("rf.csv", encoding="utf-8", index=False)
result

pd.DataFrame({
    "Name":df.columns,
    "Imp":clf.feature_importances_
}).sort_values(by="Imp", ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(y=data["Sex"], hue=data["Survived"])

bin = pd.cut(data["Fare"], bins=10)
plt.figure(figsize=(10, 10))
sns.countplot(bin, hue=data["Survived"])
plt.legend(loc="upper right")
plt.xticks(rotation=30)

# loc[列標籤], iloc[第幾列]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scale = scaler.fit_transform(df)
x_scale = df_scale[:data.shape[0]]
x_predict_scale = df_scale[data.shape[0]:]
y = np.array(data["Survived"])
pd.DataFrame(x_scale)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
params = {
    "n_neighbors":range(5, 100)
}
search = GridSearchCV(clf, params, cv=10, n_jobs=-1)
search.fit(x_scale, y)
print(search.best_score_)
print(search.best_params_)

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_scale, y)
pre = clf.predict(x_predict_scale)
result = pd.DataFrame({
    "PassengerId":predict["PassengerId"],
    "Survived":pre
})
result.to_csv("knn.csv", encoding="utf-8", index=False)
result