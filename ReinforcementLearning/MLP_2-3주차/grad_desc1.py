import numpy as np
import matplotlib.pyplot as plt

x = 10  
learning_rate = 0.2  
precision = 0.00001  
max_iterations = 100

# 손실함수를 람다식으로 정의한다. 
loss_func = lambda x: (x-3)**2 + 10

# 그래디언트를 람다식으로 정의한다. 손실함수의 1차 미분값이다. 
gradient = lambda x: 2*x-6

list1 = []
list2 = []

# 그래디언트 강하법
for i in range(max_iterations):
    x = x - learning_rate * gradient(x)
    list1.append(x)
    list2.append(loss_func(x))
    print("손실함수값(", x, ")=", loss_func(x))

print("최소값 = ", x)

x1 = np.linspace(0.0, 10.0)
y1 = loss_func(x1)
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(x1,y1)  # Plot some data on the axes.
ax.plot(list1,list2, '*')  # Plot some data on the axes.