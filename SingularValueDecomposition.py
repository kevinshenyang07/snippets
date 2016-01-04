# -*- coding:utf-8 -*-

import scipy
import matplotlib.pylab as plt

# seeding the random generate so we always get the same random numbers
np.random.seed(seed=99)

# make some data up
mean = [0,0]
cov = [[1.0,0.7],[0.7,1.0]] 
x,y = np.random.multivariate_normal(mean,cov,500).T

# plot the data
fig = plt.figure()
plt.scatter(x,y)
plt.axis('equal')
plt.show()

# create a data matrix
matrix = np.column_stack((x,y))
# compute SVD
U,s,Vh = scipy.linalg.svd(matrix)

'''
The function gives us the matrices U and Vh, but s is only returned as a vector 
In order to use s to reconstruct the data or project it down to one dimension 
we need to convert s into a diagonal matrix containing only zeros and the values
of s in the diagonal.
'''

S = scipy.linalg.diagsvd(s, 500, 2)
# reconstruct the data (sanity test)
reconstruction = np.dot(U, np.dot(S, Vh))

# check the results
print matrix[1,:]
print reconstruction[1,:]
# the allclose() function allows for the data points to deviate by a small
# epsilon and still be considered correctly reconstructed.
print np.allclose(matrix, reconstruction)

'''
In the following plot you can see the columns of V visualized along with the
data points. The thing to note is that the vectors are orthogonal to each other
and that the first vector runs along the direction of the most variance in the
data. In a way it is capturing the main source of information. These vectors
are also called the principal components of the data.
'''

# show the column vectors of V
V = Vh.T
plt.scatter(x, y)
plt.plot([0, V[0,0]], [0,V[1,0]], c='r', linewidth=10.0)
plt.plot([0, V[0,1]], [0,V[1,1]], c='y', linewidth=10.0)
plt.axis('equal')  # to make x and y axis metrics equal
plt.show()

'''
The following code shows how to project the two dimensional data points down
into one dimension. We basically project the data onto the red line in the plot
above. The second option shown also works to apply the same projection to new
data points that were generated from the same source as the data we used to
calculate the SVD.
'''

# two ways to project the data
projection = np.dot(U, S[:,:1])
projection2 = np.dot(matrix, V[:,:1])
np.allclose(projection, projection2)

'''
The following plot shows where the original two dimensional data points end up
in the projected version. What you can see is that the points at the outer ends
of the data cloud also are at the outer ends of the projected points.
'''

# compare the plots
plt.clf()  # clear the figures
zeros = np.zeros_like(projection)
plt.scatter(projection, zeros, c='r', zorder=2)
plt.scatter(x,y,c='b', zorder=2)

for px, py, proj in zip(x,y,projection):
    plt.plot([px,proj],[py,0],c='0.5', linewidth=1.0, zorder=1)
    
plt.axis('equal')
plt.show()

'''
When we calculated the SVD we did a sanity check by reconstructing the original
data from the three matrices. All we needed to do was multiply them and we got
the original data back. Now we have the points projected down to one dimension.
'''

# try to reconstruct back to 2D
# just a reminder
projection = np.dot(U, S[:,:1])
## now the reconstruction
reconstruction = np.dot(projection, Vh[:1,:])
reconstruction.shape

'''
What you can see in the plot below is that the reconstructed data still is a one
dimensional line, but now rotated in the two dimensional space of the original
data points. The loss of variance in the data actually can be a feature, because
the variance along the other dimensions might have been caused by noise in the 
data.
'''

# compare the plots
plt.clf()
# form an array of zeros with the same shape as a given array
zeros = np.zeros_like(projection) 
plt.scatter(reconstruction[:,0], reconstruction[:,1], c='r', zorder=2)
plt.scatter(x,y,c='b', zorder=2)

for px, py, rx,ry in zip(x,y,reconstruction[:,0], 
                         reconstruction[:,1]):
    plt.plot([px,rx],[py,ry],c='0.5', linewidth=1.0, zorder=1)
    
plt.axis('equal')
plt.show()