import numpy as np
import matplotlib.pyplot as plt

def step(x):
	result = x > 0.000001		# True 또는 False 
	return result.astype(np.int)	# 정수로 반환


x = np.arange(-10.0, 10.0, 0.1)
y = step(x)
plt.plot(x, y)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x)) 

x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
  
x = np.linspace(-np.pi, np.pi, 60) 
y = np.tanh(x) 
plt.plot(x, y) 
plt.show() 

import numpy as np 
import matplotlib.pyplot as plt 

def relu(x):
	return np.maximum(x, 0)
  
x = np.arange(-10.0, 10.0, 0.1)
y = relu(x)
plt.plot(x, y) 
plt.show() 