import numpy as np


#===================
#== Problem 1:
#==================


aa = np.array([1, 3])
bb = np.array([5, 2])
c = 3

# a)aa+bb
print(f"aa + bb {aa+bb} ")

# b) caa
print(f"c*aa = {c*aa} ")

# c) cbb
print(f"c*aa = {c*bb} ")

# d) c(aa+bb)
print(f"c(aa+bb) = {c*(aa+bb)} ")

# e) c(bb-aa)
print(f"c(bb-aa) = {c*(bb-aa)} ")

#===================
#== Problem 2:
#==================
print("==== Second Problem ===")


a = [1, 3]
b = [2, 6]
c = [-1, -3]
d = [1, -3]
e = [1, 2, 3]
f = [1, 2, 3, 4]

print(f"Norm {a} = {np.linalg.norm(a) } " )
print(f"Norm {b} = {np.linalg.norm(b) } " )
print(f"Norm {c} = {np.linalg.norm(c) } " )
print(f"Norm {d} = {np.linalg.norm(d) } " )
print(f"Norm {e} = {np.linalg.norm(e) } " )
print(f"Norm {f} = {np.linalg.norm(f) } " )

