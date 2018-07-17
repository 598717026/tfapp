import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from threading import *
from matplotlib.widgets import Button

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def computeYByX(x):
    noise = np.random.normal(-100, 100, x.shape)
    #return 400 * np.sin(x) + 2 * x * x #+ noise
    return 40 * x * x + 1 #+ noise


xTrain = np.linspace(-20, 20, 401).reshape([1, -1])
noise = np.random.normal(-0.2, 0.2, xTrain.shape)
yTrain = computeYByX(xTrain)

plt.clf()
plt.plot(xTrain[0], yTrain[0], 'ro', label = 'train data')
plt.legend()
plt.savefig('curve_data.png', dpi = 200)

x = tf.placeholder(tf.float32, [1, 401])


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

hiddenDim = 400

W = weight_variable([hiddenDim, 1])
b = bias_variable([hiddenDim, 1])

W2 = weight_variable([1, hiddenDim])
b2 = bias_variable([1])

W3 = weight_variable([401, 401])
b3 = bias_variable([1, 401])

hidden = tf.nn.sigmoid(tf.matmul(W, x) + b)
y = tf.matmul(W2, hidden) + b2

# y = 2 * (tf.nn.sigmoid(tf.matmul(x, W3) + b3) - 0.5)
# y = tf.matmul9x, W3) + b3

# Minimize the squared errors.
loss = tf.reduce_mean(tf.square(y - yTrain))
step = tf.Variable(0, trainable = False)
rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
optimizer = tf.train.AdamOptimizer(rate)
train = optimizer.minimize(loss, global_step = step)
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

plt.plot(xTrain[0], yTrain[0], 'ro', label = 'train data')
l, = plt.plot(xTrain[0], y.eval({x: xTrain}, sess)[0], label = 'train data')

def threadtrain():
	print("training start")
	for time in range(0, 10001):
		train.run({x: xTrain}, sess)
		if time % 1000 == 0:
			#plt.clf()
			l.set_xdata(xTrain[0])
			l.set_ydata(y.eval({x: xTrain}, sess)[0])
			plt.draw()
			#plt.legend()
			#plt.savefig('curve_fitting_' + str(int(time / 1000)) + '.png', dpi = 200)
	print("training end")

def onclickstart(event):
	Thread(target = threadtrain).start()


axnext = plt.axes([0.9,0.01,0.1,0.075])
bnext =Button(axnext,'Start')
bnext.on_clicked(onclickstart)
plt.show()

'''	
xTest = np.linspace(-40, 40, 401).reshape([1, -1])
yTest = computeYByX(xTest)

plt.clf()
plt.plot(xTest[0], yTest[0], 'mo', label = 'test data')
plt.plot(xTest[0], y.eval({x: xTest}, sess)[0], label = 'curve fitting')
plt.legend()
plt.savefig('curve_fitting_test.png', dpi = 200)
plt.show()
'''
