##########################################################
#   Name: Devon James                  Class: CSI 431
#   Date: Nov. 28th, 2018              Prof: Petkov
##########################################################

###IMPORTS###
# numeric python and plotting
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


###ACTUAL CODE###
#for x values when plotting
i = [1,2,3,4,5,6]

#Create Laplacian Matrix
L = np.matrix([
       [2,-1,-1,0,0,0],
       [-1,2,-1,0,0,0],
       [-1,-1,3,-1,0,0],
       [0,0,-1,3,-1,-1],
       [0,0,0,-1,2,-1],
       [0,0,0,-1,-1,2]
])

#Create Symmetric Laplacian Matrix
Ls = [
       [4,-2,-(np.sqrt(6)),0,0,0],
       [-2,4,-(np.sqrt(6)),0,0,0],
       [-(np.sqrt(6)),-(np.sqrt(6)),9,-3,0,0],
       [0,0,-3,9,-(np.sqrt(6)),-(np.sqrt(6))],
       [0,0,0,-(np.sqrt(6)),4,-2],
       [0,0,0,-(np.sqrt(6)),-2,4]
]


#Compute Eigen decomposition of L
lEigVal, lEigVec = la.eig(L)
#Compute Eigen decomposition of Ls
lsEigVal, lsEigVec = la.eig(Ls)


#u = eigenvector for L | uS = eigenvector for Ls
u = list(np.array(lEigVec[2]).flatten())
uS = list(np.array(lsEigVec[2]).flatten())



#PLOTTING (X = NodeID , Y = Corresp. values of u and uS)
plt.plot(i, u, label="u")
plt.plot(i, uS, label="uS")
plt.legend(loc='upper right')
plt.title('Laplacian')
plt.xlabel('NodeID')
plt.ylabel('u and uS')
plt.show()
