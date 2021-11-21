
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns

# x_value = np.arange(0 , 10 , 0.01 )
# y_value = np.sin(2*np.pi* x_value)
# plt.plot(x_value , y_value)
# plt.grid()
# ticks_x = np.linspace(0, 10, 4)
# months = ['Jan','Feb','Mar','Apr']
# plt.xticks(ticks_x, months, fontsize = '20', family = 'fantasy',color='orange')
# # plt.xticks(np.arange(0,12,2))
# plt.yticks( [1,-1], ["Peak" ,"Valley"])
# plt.title("Sine Line Char" , fontsize = "20" , weight = "bold")
# plt.ylabel("Y-Axis" , fontsize = 20 , color = "blue")
# plt.show()


# data_list = [1,2,1,3]
# data_list_2 = [3, 2.3, 2, 0]
# data_list_3 = [2, 4.3, 1, 0]
# plt.plot(data_list , c = "blue" , linewidth = 3)
# plt.xticks
# plt.plot(data_list_2)
# plt.plot(data_list_3, linestyle = '-.',linewidth = 5, c = 'r')
# plt.legend(['First Line','Second Line'],shadow = True, loc = 0)
# plt.show()

# new_x = np.arange(1,4)
# a = (new_x[0] / np.sum(new_x)  )*100
# b = (new_x[1] / np.sum(new_x)  )*100
# c = (new_x[2] / np.sum(new_x)  )*100
# plt.pie(new_x,labels=[f"Babase {a} " , b ,c] , autopct = "%.f" )
# plt.show()



new_x = np.arange(20)
new_y = np.random.randint(5,10,20)

# plt.figure(figsize=[10,6])
# plt.suptitle('Main Title',fontsize = 16)

# plt.subplot(2,2,1) # (number of rows, number of columns, number of plot)
# plt.plot(new_x, new_y)
# plt.title('Line Plot')
#
# plt.plot(new_x, new_y)
# plt.title('Line Plot')
#
# plt.scatter(new_x, new_y)
# plt.title('Line Plot')
#
# plt.subplot(2,2,4)
# plt.bar(new_x, new_y)
# plt.title('Bar Chart')
#
# plt.show()


# let's load the data and store it in a Pandas DataFrame
iris = datasets.load_iris()
data = pd.DataFrame(iris.data[:, :4],columns = iris.feature_names)
data['target'] = iris.target # 0-'setosa', 1-'versicolor', 2-'virginica'
data.target.replace(to_replace=[0,1,2], value=iris.target_names, inplace=True)

print(data.head())
# sns.pairplot(data)
sns.pairplot(data , hue = "target" , diag_kind = "kde")
plt.show()





