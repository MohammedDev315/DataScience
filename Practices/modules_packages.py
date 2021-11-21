import math # this is how we import a module

import random

dir(math)
math.sin(0)
math.sqrt(9)
random.randint(0,9) # Return random integer in range [a, b], including both end points.

values = [1, 4, 9, 16, 25, 36]
number1 = random.choice(values) # Return random form array
print(number1)
number2 = random.sample(values, 2)
print(number2)


# For String========
# Methods are a specific type of functions that belong to an object. If we type a string followed by a . and then a TAB we can observe the existing methods.

text = "Hello World"

text.index('o') # index where the first "o" occurs
dir(text)
text = "Hello World"
tex_res = text.count('l') # counts the number of lowercase 'l'
print(tex_res)
text = "Hello World"
tex_res = text.upper() # what do you think text.lower() will do. Also, it is not in-place!
print(tex_res)
# We can also split a string into a list.
text = "Hello World"
tex_res = text.split() # the default is to split by the empty space
print(f"Split func {tex_res} ")
# We can also split by any character.
text = "Hello World"
tex_res = text.split('o')
print(f"Split with o : {tex_res}")
# We can convert a list into a string using "".join(list_name)
text = ["Hello ","World ","and ","Universe"]
tex_res = text2 = "".join(text) #join only works with String
print(f"Join all element of an array : {tex_res} ")


#------------------------------------------
# remove - from number and return it as int
text = "1-800-123-1234"
def converter(text):
    text = text.split('-')
    return int(''.join(text)) # convert text after join to int so you used cast

# is... methods¶
# There are a few methods that allows us to determine if the string is an alphabet (.isalpha), number (.isnumeric), lowercase (.islower), etc.
# Write a function that receives a string and returns three strings. One string has all the lowercase alphabet characters, the other has all the numerical characters, and the thirds string has all the rest. Note any other character or space is ignored.
# For example, ab,c1K2H3de f456! should return abcdef, 123456 and ,KH !
text = "ab,c1K2H3de f456!"

def alpha_num_fun(text):
    alpha = ""
    num = ""
    other = ""
    for c in text:
        if c.isalpha() and c.islower():
            alpha +=c
        elif c.isnumeric():
            num +=c
        else:
            other += c
    return alpha, num, other

alpha, num, other = alpha_num_fun(text)
print(f"Origin text : {text}")
print(f"The result : {alpha} {num }  {other}")



# Replace
# We can also replace sub-strings with .replace()

text = ("How much wood could a woodchuck chuck "
        "If a woodchuck could chuck wood? ")

text = text.replace("wood", "cheese")
print(f"this is result after replaceing : {text} ")







