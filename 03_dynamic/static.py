import tensorflow as tf
import numpy as np

# Set up the computational graph

N = 64
D_in = 1000
H = 100
D_out = 10

# Placeholders for input and target data
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for weights
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss
loss = tf.reduce_sum((y - y_pred) ** 2)

# Compute gradient of loss
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Enter a TensorFlow session to execute
with tf.Session() as sess:
    # Run graph once to initialize w1 and w2
    sess.run(tf.global_variables_initializer())

    # Create numpy arrays holding the data for x and y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for _ in range(500):
        # Execute graph many times
        # Binding x_value to x
        # and     y_value to y
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
        print(loss_value)
