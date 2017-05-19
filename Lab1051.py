from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# create 2 clusters of 25 data samples. THe two clusters can be identified by their different mean.
mean1 = [-4, -4]
mean2 = [3, 3]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean1, cov, 25).T
x1, y1 = np.random.multivariate_normal(mean2, cov, 25).T
X = np.concatenate([x, x1])
Y = np.concatenate([y, y1])

#The true amount of clusters is two, but you can try to adjust the k value.
k=2

# using k++ for initialization and run 1 time. THis is just to show that the next kmeans were we run it 20 times 
 # should minimize the sum of squared distances more
kmeans = KMeans(algorithm='auto', n_clusters=k, init='k-means++', n_init=1).fit(zip(X,Y))
print "sum of squared distance at n_init=1: "
print kmeans.inertia_

#using k++ for initialization and run 20 times and take the centroids that minimizes the sum
kmeans = KMeans(algorithm='auto', n_clusters=k, init='k-means++', n_init=20).fit(zip(X,Y))
labels = kmeans.labels_ #labels for our data points
centroids = kmeans.cluster_centers_
print "sum of squared distance at n_init=20: " 
print kmeans.inertia_

#plot centroids
lines = plt.plot(centroids[:,0], centroids[:,1], 'kx')
plt.setp(lines, ms=15.0)
plt.setp(lines,mew=2.0)

#PLot observations belonging to different clusters with different color
for i in range(k):
    dsx = X[np.where(labels==i)]
    dsy = Y[np.where(labels==i)]
    #plt.plot(np.zeros(len(ds)), ds,'o')    
    plt.plot(dsx, dsy, 'o')
plt.show()