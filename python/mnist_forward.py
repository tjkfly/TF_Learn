import  tensorflow as tf
import os
# from tensorflow import keras
# from tensorflow.keras import datasets
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32) /255.
y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32) /255.
y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
print("hello")

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)

w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros(256))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros(128))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros(10))
lr = 1e-3

for epoch in range(20):
    for step,(x,y) in enumerate(train_data):
        x = tf.reshape(x,[-1,28*28])
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            h3 = h2 @ w3 + b3
            y = tf.one_hot(y,depth=10)
            loss = tf.reduce_mean(tf.square(h3-y))
        grade = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])

        w1.assign_sub(lr * grade[0])
        b1.assign_sub(lr * grade[1])
        w2.assign_sub(lr * grade[2])
        b2.assign_sub(lr * grade[3])
        w3.assign_sub(lr * grade[4])
        b3.assign_sub(lr * grade[5])
        if step % 100 == 0:
            print(epoch,step,"loss:",float(loss))
    total_correct = 0
    total_num = 0
    for step,(x,y) in enumerate(test_data):
        x = tf.reshape(x,[-1,28*28])
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        h3 = h2 @ w3 + b3
        prob = tf.nn.softmax(h3,axis=1)

        pred = tf.argmax(prob,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)

        correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct /total_num
    print("ACC = ",acc)