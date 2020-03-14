import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x,y):
	x = tf.cast(x,dtype=tf.float32) / 255.
	y = tf.cast(y,dtype=tf.int32)
	return x,y



(x_train, y_train),(x_test, y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape,y_train.shape)

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.map(preprocess).shuffle(10000).batch(128)

test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data = test_data.map(preprocess).batch(128)


model = Sequential([
	layers.Dense(256,activation=tf.nn.relu),  #[b,784] [b,256]
	layers.Dense(128,activation=tf.nn.relu),  #[b,256] [b,128]
	layers.Dense(64,activation=tf.nn.relu),
	layers.Dense(32,activation=tf.nn.relu),
	layers.Dense(10)
])
model.build(input_shape=[None,28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)
def main():
	for epoch in range(10):
		for step,(x,y) in enumerate(train_data):
			x = tf.reshape(x,[-1,28*28])
			with tf.GradientTape() as tape:
				logits = model(x)
				y_onthot = tf.one_hot(y, depth=10)
				loss_mse = tf.reduce_mean(tf.losses.MSE(y_onthot,logits))
				loss_ce = tf.losses.categorical_crossentropy(y_onthot,logits,from_logits=True)
				loss_ce = tf.reduce_mean(loss_ce)
			grads = tape.gradient(loss_ce,model.trainable_variables)
			optimizer.apply_gradients(zip(grads,model.trainable_variables))
			if step % 100 ==0:
				print(epoch,step,float(loss_mse),float(loss_ce))
		total_correct = 0
		total_num = 0
		for x,y in test_data:
			x = tf.reshape(x,[-1,28*28])
			logits = model(x)
			prob = tf.nn.softmax(logits,axis=1)
			pred = tf.argmax(prob,axis=1)
			pred = tf.cast(pred,dtype=tf.int32)
			correct = tf.equal(pred,y)
			correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))

			total_correct += int(correct)
			total_num += x.shape[0]
		acc = total_correct /total_num
		print(epoch,"test_acc",acc)








if __name__ == '__main__':
	main()