import numpy as np
import matplotlib.pyplot as plt


# data_list = [1,2,1,3,5,6,3,4,5,1,3,6]
# data_list_2 = [1,1,5,3,4,6,6,5,5,3,4,3]
# plt.plot(data_list , linewidth = 2)
# plt.plot(data_list_2)
# plt.legend(['First Line','Second Line'],shadow = True, loc = 0)
#
#
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.title("test plot")




new_x = np.arange(20)
new_y = np.random.randint(5,10,20)

plt.figure(figsize=[15,5])

plt.suptitle('Main Title',fontsize = 16)

plt.subplot(1,2,1) # (number of rows, number of columns, number of plot)
plt.plot(new_x,new_y)
plt.title('Line Chart')

plt.subplot(1,2,2)
plt.bar(new_x,new_y)
plt.title('Bar Chart');


plt.show(block=True)
plt.interactive(False)






