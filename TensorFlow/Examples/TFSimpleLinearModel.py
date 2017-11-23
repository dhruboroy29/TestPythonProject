import tensorflow as tf

# Initialize weight variable (trainable)
W = tf.Variable([0.3], dtype=tf.float32)

# Initialize bias variable (trainable)
b = tf.Variable([-0.3], dtype=tf.float32)

# Initialize input placeholder
x = tf.placeholder(tf.float32)

# Define model
simple_linear_model = W * x + b

# Start session
sess = tf.Session()

# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)

# Print weight and bias
print("[W], [b]: ", W.eval(session=sess), b.eval(session=sess))

# Evaluate linear model
print(sess.run(simple_linear_model, {x: [1, 2, 3, 4]}))

# Now, define loss function
