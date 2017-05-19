from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial import distance 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# create 2 clusters of 25 data samples. THe two clusters can be identified by their different mean.
mean1 = [-4, -4]
mean2 = [3, 3]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean1, cov, 25).T
x1, y1 = np.random.multivariate_normal(mean2, cov, 25).T
X = np.concatenate([x, x1])
Y = np.concatenate([y, y1])

#Using scaling before clustering
scaled_x = scale(X)
scaled_y = scale(Y)
z_complete = (linkage(zip(scaled_x,scaled_y), method='complete', metric='euclidean'))
#z_complete = linkage(zip(X,Y), method='complete', metric='euclidean')
z_average = linkage(zip(X,Y), method='average', metric='euclidean')
z_single = linkage(zip(X,Y), method='single', metric='euclidean')

print cut_tree(z_complete, n_clusters=2).T
print cut_tree(z_average, n_clusters=2).T
print cut_tree(z_single, n_clusters=2).T
print cut_tree(z_single, n_clusters=4).T

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    z_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)


# create 2 clusters of 25 data samples. THe two clusters can be identified by their different mean.
mean = [0, 0]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean, cov, 25).T
x1, y1 = np.random.multivariate_normal(mean, cov, 25).T
x2, y2 = np.random.multivariate_normal(mean, cov, 25).T
X = np.concatenate([x, x1, x2])
Y = np.concatenate([y, y1, y2])


dd = distance.pdist(1-np.corrcoef(zip(X,Y)))
z_complete = linkage(dd, method='complete', metric='euclidean')
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    z_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
