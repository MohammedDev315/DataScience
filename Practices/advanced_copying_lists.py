#insert a list as list
x = [1, 2, 3]
y = ["a", "b"]
x.append(y)
print(x)

#insert  items of two lists  together
x = [1, 2, 3]
y = ["a", "b"]
print(x+y)
x.extend(y)
print(x)
x = [1, 2, 3]
y = ["a", "b"]
x += y
print(x)


# Write a for loop that prints the maximum value of a list.
my_list = [33, 5, 9, 14, 57, -3, 22] # should print 57
curre_number = 0
for x in my_list:
    if curre_number < x:
        curre_number = x
print(f"Maxumum Value is {curre_number} ")


# List Methods
# Below we present a few methods that are commonly used on lists.
x = [1, 20, 3, 20]
x.index(20) # index where the value occurs for the first time

x = [1, 20, 3, 20]
x.count(20)

x = [1, 20, 3, 20]
x.reverse()
print(x)



# Sorting
# We can sort lists in two ways:
# Using a sorted function which does not modify the list (not in place)
# Using a sort method which modifies the list (in place)

# Not in place
x = [1, 20, 3, 20]
print(sorted(x))
print(x)

# In place => modifies the list
x = [1, 20, 3, 20]
x.sort()
print(x)



# Shallow Copy¶
x = [1,2,3]
y = x.copy() # you can also do y = x[:]
print(f"x is {x}")
print(f"y is {y}")


# Let's modify x
x[0] = "a"
print(f"x is {x}")
print(f"y is {y}")


# Given that we used the copy method we were able to obtain the expected behavior. This is known as a shallow copy. The reason it contains the word shallow can be explained in the next example.
x = [[0,1],2,3]
y = x.copy() # you can also do y = x[:]
print(f"x is {x}")
print(f"y is {y}")
x[0][1] = "a"
x[1] = "b"
print(f"x is {x}")
print(f"y is {y}")




# As we notice, the inner list still has the behavior of the view. We have one more option to obtain a behavior in which the two lists are independent. For this we use a deep copy.
# Deep Copy
import copy # we are importing a library called copy.  We will talk about libraries in a future lecture.
x = [[0,1],2,3]
y = copy.deepcopy(x)
print(f"x is {x}")
print(f"y is {y}")

x[0][1] = "a"
x[1] = "b"

print(f"x is {x}")
print(f"y is {y}")




