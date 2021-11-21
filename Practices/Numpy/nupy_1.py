import numpy as np
import numpy.matlib

v1 = np.array([ [3 , 5 , 4 , 9 ] , [2 ,4 , 4 , 3 ] ] )
v2 = v1.reshape(4,2)
np_list = np.arange(10)
v3 = np_list.reshape(2,5) # Must be equle the length of list or array
v4 = np.empty([3,3] , dtype=int)
py_list = [3,5,1,2]
v5 = np.asarray(py_list) #useful for converting Python List into Numpy Array
v6 = np.array(py_list) # this is also works
v7 = np.asarray(py_list , dtype=float) #covert list to float
py_tuple = (6 , 8 , 4 , 5)
v8 = np.asarray(py_tuple)
np_list2 = np.arange(10 , 100, 14 ) # numpy.arange(start, stop, step, dtype)
v9  = np.linspace(10 , 40 , 6 , dtype=int) # base on third element. it will be number of elements
v10 = np.logspace(1,10,num = 10, base = 2 , dtype=int)  # First result will be base sequr first elem. = 2^1
np_list3 = np.arange(1 , 20)
v11 = np_list3[slice(1,20,3)] #This slice object is passed to the array to extract a part of array.
# Target specifi postion in a array:
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols] #first reslut = x[0][0] , second = x[0][2] .....ets
# # Iterating Over Array and git its data.
# a = np.arange(0, 20, 5)
# a = a.reshape(2,2)
# b = a.T # this will re-order , optional
# for x in np.nditer(b):
#    print(x)
# for x in np.nditer(a, flags = ['external_loop'], order = 'F'): # order is  optional
#    print(x),
# # Broadcasting Iteration
# # If two arrays are broadcastable, a combined nditer object is able to iterate upon
# # them concurrently. Assuming that an array a has dimension 3X4, and there is another
# # array b of dimension 1X4, the iterator of following type is used
# # (array b is broadcast to size of a).
# a = np.arange(0,60,5)
# a = a.reshape(3,4)
# print ('First array is:' )
# print (a)
# print ('Second array is:')
# b = np.array([1, 2, 3, 4], dtype = int)
# print (b)
# print ('Modified array is:' )
# for x,y in np.nditer([a,b]):
#    print("%d:%d" % (x,y))

v12 = np.arange(8).reshape(2,4)
flat_v12 = v12.flatten(order = 'F') #this will make v12 one dimantion array
v13 = np.linspace(1,30 , num=12 , dtype=int ).reshape(2,6)
change_col_row_v13 = np.transpose(v13) #permutes the dimension of the given array
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])
b = np.broadcast(x, y)


arr_join1 = np.array([[1,2,3] , [4,5,6]])
arr_join2= np.array([[7,8,9] , [10,11,12]])
#This will change array dimnation:
coc_1 = np.concatenate((arr_join1 , arr_join2) , axis= 0)#will add arr1 under arr2 => 4*3
coc_2 = np.concatenate((arr_join1 , arr_join2) , axis= 1)#will add first row for each arr togater => 2*6
#return same array dimantion but different order
stack_1 = np.stack((arr_join1 , arr_join2 ) , axis=0) #will added first arr. under seocd arr
stack_2 = np.stack((arr_join1 , arr_join2 ) , axis=1) #will add first row for each arr.

arr_spl1 = np.arange(16).reshape(4,4)
split1 = np.split(arr_spl1 , 4) # must be same as arr. dimantion
split2 = np.split(arr_spl1 , [1,2]) # it not clear enuogh
split_h = np.hsplit(arr_spl1 , 2) #number => how many colunm in each row : dimation impotant
split_v = np.vsplit(arr_spl1 , 2) #number => how many row in each arr : dimation impotant


# This function returns a new array with the specified size. If the new size is greater than
# the original, the repeated copies of entries in the original are contained.
# numpy.resize(arr, shape) => np.resize( arr , () )
arr_reshape = np.arange(16).reshape(4*4)
re_size = np.resize(arr_reshape , (4,5) )

# This function adds values at the end of an input array. The append
# operation is not inplace, a new array is allocated. Also the dimensions of
# the input arrays must match otherwise ValueError will be generated.
a = np.array([[1,2,3],[4,5,6]])
np.append(a, [7,8,9]) # will be one dimnation arr.
np.append(a, [[7,8,9]],axis = 0) # Columns will not change, row changed
np.append(a, [[5,5,5],[7,8,9]],axis = 1) # row not change, columns changed


# This function inserts values in the input array along the given axis and before
# the given index. If the type of values is converted to be inserted, it is different
# from the input array. Insertion is not done in place and the function returns a new
# array. Also, if the axis is not mentioned, the input array is flattened.
a = np.array([[1,2],[3,4],[5,6]])
np.insert(a,3,[11,12]) # return => one dimation arr #3 is the postion where arr. will added
np.insert(a,1,[11],axis = 0)  # will added to 1 postion, [11] will repeted becuse orginal arra is 2 columns
np.insert(a,1,[11],axis = 1) # 11 will added to all 1 postion at each array


a = np.arange(12).reshape(3,4)
np.delete(a,5)
np.delete(a,1,axis = 1)
# a = np.array([1,2,3,4,5,6,7,8,9,10])
np.delete(a, np.s_[::2] , axis=1)

#function For String
np.char.center('Second', 20,fillchar = '*')
np.char.split ('hello how are you?')
np.char.split ('TutorialsPoint,Hyderabad,Telangana', sep = ',')
np.char.join(':','dmy')
np.char.join([':','-'],['dmy','ymd'])
np.char.replace ('He is a good boy', 'is', 'was')


a = np.array([1.0,5.55, 123, 0.567, 25.532])
np.around(a) # => [  1.   6. 123.   1.  26.]
np.around(a, decimals = 1) # => [  1.    5.6  123.    0.6  25.5]
np.around(a, decimals = -1) # => [  0.  10. 120.   0.  30.]

a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
np.ceil(a) # => [-1.  2. -0.  1. 10.]

a = np.arange(9, dtype = np.float_).reshape(3,3)
b = np.array([10,10,10])
np.add(a,b)
np.subtract(a,b)
np.multiply(a,b)
np.divide(a,b)

a = np.array([10,100,1000])
np.power(a,2)
b = np.array([1,2,3])
np.power(a,b)

a = np.array([10,20,30])
b = np.array([3,5,7])
np.mod(a,b)
np.remainder(a,b)



a = np.array([[3,7],[9,1]])
np.sort(a)
np.sort(a, axis = 0)
dt = np.dtype([('name', 'S10'),('age', int)])
a = np.array([("raju",21),("anil",25),("ravi", 17), ("amar",27)], dtype = dt)
np.sort(a, order = 'name')
#using Where :
# y = np.where(a > 3)
# a[y]
#using extrict :
# condition = np.mod(a,2) == 0
# np.extract(condition, a)
# 'Create a deep copy of a:'
# b = a.copy()


np.matlib.rand(3,3)
np.linspace(1 , 33 , num=9 ).reshape(3,3)
np.matlib.zeros((2,2))
np.matlib.ones((2,2))
i = np.matrix('1,2;3,4')


# NumPy - Linear Algebra
#=======================

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
np.dot(a,b)
# returns the dot product of the two vector
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
np.vdot(a,b)
# returns the inner product of vectors for 1-D arrays. For higher
# dimensions, it returns the sum product over the last axes.
np.inner(np.array([1,2,3]),np.array([0,1,0]))
# Equates to 1*0+2*1+3*0
a = np.array([[1,2], [3,4]])
b = np.array([[11, 12], [13, 14]])
# 1*11+2*12, 1*13+2*14
# 3*11+4*12, 3*13+4*14
np.inner(a,b)



