import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_excel("./Summary of main results.xlsx", skiprows=1)
X = df[['Annual Surface Solar Radiation (Kwh/m2)', 'Rooftop Area (km2)', 'Grid Emission Factor (gCO2/Kwh)']].values
# standardization
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=1212).fit(X_std)
classes = kmeans.predict(X_std)
df["pred"] = classes
df["pred"] = df["pred"].map({0:2, 1:1, 2:3, 3:4})
df.to_excel("./cluster_results.xlsx", index=None)