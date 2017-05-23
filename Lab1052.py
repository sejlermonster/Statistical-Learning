from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial import distance 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# create 2 clusters of 25 data samples. THe two clusters can be identified by their different mean.
# Generate data
np.random.seed(3)
X = np.random.standard_normal((50,2))
X[:25,0] = X[:25,0]+3
X[:25,1] = X[:25,1]-4

#Using scaling before clustering
#scaled_x = scale(X)
z_complete = (linkage(scaled_x, method='complete', metric='euclidean'))
z_complete = linkage(X, method='complete', metric='euclidean')
z_average = linkage(X, method='average', metric='euclidean')
z_single = linkage(X, method='single', metric='euclidean')

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
    z_single,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)


# # create 2 clusters of 25 data samples. THe two clusters can be identified by their different mean.
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x, y = np.random.multivariate_normal(mean, cov, 25).T
# x1, y1 = np.random.multivariate_normal(mean, cov, 25).T
# x2, y2 = np.random.multivariate_normal(mean, cov, 25).T
# X = np.concatenate([x, x1, x2])
# Y = np.concatenate([y, y1, y2])


# dd = distance.pdist(1-np.corrcoef(zip(X,Y)))
# z_complete = linkage(dd, method='complete', metric='euclidean')
# # calculate full dendrogram
# plt.figure(figsize=(25, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(
#     z_complete,
#     leaf_rotation=90.,  # rotates the x axis labels
#     leaf_font_size=8.,  # font size for the x axis labels
# )
# plt.show()
