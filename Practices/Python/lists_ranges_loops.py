# Given a list,
# if the second element (index of 1) is greater than the first element (index of 0), and
# if the third element is the letter 'a', and
# the last element is of type float,
# then print This is a valid list;
# otherwise, print This is NOT a valid list

# These examples should print `This is a valid list`
test_list = [1, 2, 'a', 3.3]
# These examples should print `This is NOT a valid list`
# test_list = [2, 1, 'a', 3.3]
# test_list = [1, 2, 'b', 5, 3.3]
# test_list = [2, 1, 'a']
# test_list = [1, 2, 'b', 5]

if test_list[1] > test_list[0] and test_list[2] == 'a' and isinstance(test_list[-1], float):
    print("This is a valid list")
else:
    print("NOT a valid list")

# Range is a function in Python that returns a range object. We can use this to create lists that contain values between a start and end point.
a = list(range(1, 20, 2))
print(a)


# Exercise
# Create a list that contains the values from 0 to 100 (inclusive) using range. Then use list slicing to extract:
# Every fifth element (ie. [0, 5, 10, ..., 100])
# Every fifth element in reverse (ie. [100, 95, 90, ..., 0])
my_list = list(range(101))
print(my_list[::5])
print(my_list[::-5])


# Write a for loop that calculates the sum of all the elements in the list. The result should be 45.
my_list = [10, 11, 8, 5, 2, 9]

sum_res = 0
for val in my_list:
    sum_res += val
print(sum_res)



# The built-in enumerate function introduces a third way of doing for loops -- looping both by content and by index.

original_list = ['a','b','c']
for idx in range(len(original_list)):
    print(f"For index {idx} the value is {original_list[idx]}")


original_list = ['a','b','c']
for idx, val in enumerate(original_list):
    print(f"For index {idx} the value is {val}")

# Note that this is a method which means we have the list name, then add a period . and when we press TAB we get a few options of things we can do with lists.
x = ["a", "b", "c"]
print(x)
x.append("d")
print(x)

#-----------------

x = ["a", "b", "c"]
print(x)
x.insert(1,"a2")
print(x)








