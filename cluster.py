import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score


df = pd.read_csv('cleaned.csv')
df = df.dropna(subset=['Geo Local Area', 'area', 'bedroom', 'bathroom', 'built_year', 'garage', 'total_value', 'unit_price', 'change_rate'])
X = df[['Geo Local Area', 'area', 'bedroom', 'bathroom', 'built_year', 'garage', 'total_value', 'unit_price', 'change_rate']]
X = pd.concat([X.drop(columns=['Geo Local Area']), pd.get_dummies(X['Geo Local Area'])], axis=1)
X = X.fillna(0)

# k means determine k using elbow graph
distances = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distances.append(kmeanModel.inertia_)

# Plot the elbow
plt.figure(figsize=(5,5))
plt.plot(K, distances, 'bo-', alpha=0.5, c='green', lw=5)
plt.xlabel('# of clusters')
plt.ylabel('Distortion')
plt.grid(True)
plt.show()

# try build kmeans graph
range_n_clusters = [6, 7, 8, 9, 10, 11]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    silhouette_avg = silhouette_score(X, y_pred)
    sample_silhouette_values = silhouette_samples(X, y_pred)

    y_lower = 10
    cmap = cm.get_cmap("nipy_spectral")
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples


    ax1.set_title("Silhouette plot for " + str(n_clusters) + " clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    X['cluster'] = kmeans.labels_
    df1 = X.groupby(['cluster']).mean()
    # df1 = X.groupby(['cluster']).agg({'total_value':['count', 'mean'], 'unit_price':'mean', 'change_rate':'mean', 'area':'mean', 'bedroom':'mean', 'bathroom': 'mean'})
    df1.to_csv(f'{n_clusters} clusters.csv')