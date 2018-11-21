import numpy as np
from random import random
####
# Numpy tutorial : https://www.youtube.com/watch?v=EEUXKG97YRw
####

###########################
#Ufunc : universal function
###########################

#Example 1

#List
a = [1,2,3,4,5]
b = [val + 5 for val in a]
print(b)

#Numpy
a = np.array([1,2,3,4,5])
b = a + 5 #element-wise addition

print(b)

# all arithmetic operators are element wise with numpy


#############################
#Aggregations
############################

#List
c = [random() for in in range(100000)]
m = min(c)

#Numpy
c = np.array(c)
c.min()

#Matrix operation
M = np.random.randint(0,10, (3,5)) # 3 colums, 5 row matrix with values 0 to 10
M.sum()
M.sum(axis=0) #sum of all colums
M.sum(axis=1) #sum of all rows

#############################
#Broadcasting
############################

#Rules:
#1. if array shapes differ left pad the smaller shape with ones
#2. if any dimensions does not match, broadcasting the dimensions with size 1
#3. if nither non-matching dimension is 1  raise error

#############################
# Slicing masking and fancy indexing
############################

#List
L = [1,2,3,4,5]
L[0] # 2
l[1:3] # [2,3]

#Numpy

#Mask
L = num.array(L)
mask = (L>2) | (L<4)
L[mask] # [3]

#indexing
ind = [0,2,4]
L[ind]

# Matrix
M = np.arange(6).reshape(2,3)
M[:, 1] # All rows and colum 1
