import numpy as np


#
#
# arr1 = np.array([2,3,5,1,5,3,5])
# arr2 = np.array([[4,3,6] , [4,2,7] , [2,6,2] , [9,6,7]])
# arr3 = np.array([1,6,3])
#
# # print(arr2[arr2 > 3])
# # print(np.sum( arr1[arr1 > 3] ))
#
# a = np.array([10,20,30,40,50])
# print(a[[1,0,2,3]])

#
# # Matrix Multiplication
# mat_a = np.random.uniform(size=(2, 4))
# mat_b = np.random.uniform(size=(4, 3))

# print(mat_a)
# print("=====")
# print(mat_b)
# print("=====")
# print(mat_a@mat_b)
# print("======")
# print(np.dot(mat_a, mat_b))

#
# a = np.array([1,2,3])
# b = np.array([4,5,6])

# print(np.dot(a,b))

#
# print(np.random.uniform(1,50,5))
#
# arr1 = np.arange(1,4,1)
# arr2 = np.linspace(1,14,6 , dtype = int)

# print(arr1)
# print(arr2)


#
#
# arr_join1 = np.array([[1,2,3] , [4,5,6]])
# arr_join2= np.array([[7,8,9] , [10,11,12]])
# #This will change array dimnation:
# coc_1 = np.concatenate((arr_join1 , arr_join2) , axis= 0)#will add arr1 under arr2 => 4*3
# coc_2 = np.concatenate((arr_join1 , arr_join2) , axis= 1)#will add first row for each arr togater => 2*6
# #return same array dimantion but different order
# stack_1 = np.stack((arr_join1 , arr_join2 ) , axis=0) #will added first arr. under seocd arr
# stack_2 = np.stack((arr_join1 , arr_join2 ) , axis=1) #will add first row for each arr.
# print(stack_1)
# print("===")
# print(stack_2)
# print("===")
# print(coc_1)

#
#
# a = np.arange(9, dtype = np.float_).reshape(3,3)
# b = np.array([10,10,10])
# np.add(a,b)
# np.subtract(a,b)
# np.multiply(a,b)
# np.divide(a,b)
#
#
# np.mod(a,b)
# np.remainder(a,b)

#
#
# arr4 = np.linspace(1,20,8 , dtype=int)
# # print(arr4)
# print(arr4[2:6:1])
# print(arr4[2:6:2])
# print(arr4[::-1])print(arr4)
# print(arr4[2:6:1])
# print(arr4[2:6:2])
# print(arr4[::-1])

#
# arr_rand1 = np.random.randint(1,5,8) #(stat , end , total number returned)
# arr_rand2 = np.random.randint(1,5, size=(2,4)) #(stat , end , return a matrax of (row,c) )
# arr_rand3 = np.random.randint(1,5, size=(4,3)) #(stat , end , return a matrax of (row,c) )
# print(np.dot(arr_rand2 , arr_rand3))
# print(np.multiply(arr_rand2 ,3))

# arr1 = np.random.randint(1,20 , 8)
# arr2 = np.random.randint(1,30 , size=(3,4))
# res1_arr1 = arr1[arr1 > 6]
# res2_arr1 = arr1[arr1 % 2 != 0]
# print(arr2)
# res1_arr2 = arr2[arr2 > 5]
# print(arr2[0:2 , 1]) # get tow first rows and second column(its index is 1)
# res2_arr2 = arr2[arr2[0:3 , 1 ] > 13]
# res3_arr2 = arr2 + 100
# res4_arr2 = arr2[ (arr2 > 20) | (arr2 < 30) ]
# res5_arr2 = arr2[ (arr2 > 20) & (arr2 < 30) ]
# print("=====")
# print(res5_arr2)



arr1 = np.random.randint(1,20 , size=(4,5))
print(arr1)
arr1[: , 1 ] = [5,4,4,4] # change value of column => new value length must be same as rows' number
print(arr1)
print(arr1[: , 1:3])
arr1[: , 1:3] = arr1[: , 1:3] + 50
print(arr1)


