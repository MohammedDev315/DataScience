import numpy as np
import matplotlib.pyplot as plt


soa = np.array([
[1, 2],
[2, 2],
[-1, 3],
[3, -2],
[1, 0]
])

X, Y = zip(*soa)
plt.figure()
ax = plt.gca()
ax.quiver(X, Y,  angles='xy', scale_units='xy', scale=1, color = ["blue", "red" , "green"  , "yellow" , "black"])
ax.set_xlim([-5, 8 ])
ax.set_ylim([-5, 8 ])
plt.title('Dot Product (a = blue, b = red , c = green , d = yellow , f = black)')
plt.xlabel('$x_1$', fontsize = 20)
plt.ylabel('$x_2$', fontsize = 20)
plt.grid()


plt.show(block=True)
plt.interactive(False)





