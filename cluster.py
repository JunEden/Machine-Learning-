import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
# Series: df["sepal length (cm)"]
df

import numpy as np
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3)
cluster.fit(np.array(iris["data"]))

labels = cluster.labels_
df["label"] = labels
df

cluster.cluster_centers_

from sklearn.metrics import silhouette_score
for testk in range(2, 10):
    cluster = KMeans(n_clusters=testk)
    cluster.fit(iris["data"])
    score = silhouette_score(iris["data"], cluster.labels_)
    print(testk, score)

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x=df["sepal length (cm)"],
                y=df["petal length (cm)"],
                hue=iris["target"])