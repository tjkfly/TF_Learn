import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
def himmelblau(x):
    return (x[0]**2 + x[1]-11) **2 +(x[0] + x[1]**2 -7)**2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y',x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print('X,Y',X.shape,Y.shape)
Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_xlabel('y')
plt.show()

x = tf.Variable([-4.,0.])

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grad = tape.gradient(y, [x])
#     x.assign_sub(0.01 * grad[0])
    x.assign_sub(0.01 * grad[0])
    if step % 20 == 0:
         print('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
