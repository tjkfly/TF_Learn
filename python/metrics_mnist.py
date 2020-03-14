import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

def preprocess(x,y):
    x = tf.cast(x,tf.float32) / 255.
    y = tf.cast(y,tf.int32)
    return x,y




#导入数据集
(x_train, y_train),(x_test,y_test) = datasets.mnist.load_data()
print("训练集维度",x_train.shape,y_train.shape)
print("测试集维度",x_test.shape,y_test.shape)
print("当前类型：",type(x_train))
# 转为tensor
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.map(preprocess).shuffle(10000).batch(128).repeat(10)

test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data = test_data.map(preprocess).batch(128)


model = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=[None,28*28])
model.summary()

optimizer = optimizers.Adam(lr=0.01)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for step ,(x, y) in enumerate(train_data):
    with tf.GradientTape() as tape:
        x = tf.reshape(x,[-1,28*28])
        out = model(x) # [b,10]
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,out,from_logits=True))

        loss_meter.update_state(loss)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    if step % 100 == 0:
        print(step,'loss',loss_meter.result().numpy())
        loss_meter.reset_states()
    if step % 500 == 0:
        total = 0
        total_correct = 0
        for step,(x,y) in enumerate(test_data):
            x = tf.reshape(x,[-1,28*28])
            out = model(x) #[b,10]
            pred = tf.nn.softmax(out,axis=1)  #[b]
            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred,tf.int32)
            #自己的acc
            correct = tf.equal(pred,y)
            correct = tf.cast(correct, tf.int32)

            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total += x.shape[0]
            #keras 的 acc
            acc_meter.update_state(y,pred)
        print(step,'Evaluate Acc:',total_correct/total,acc_meter.result().numpy())
















# if __name__ =='__main__':
# 	main()

