# -*- coding: utf-8 -*-

#Tensorflow在mnist数据集上实现Alexnet
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

#这里可以通过tensorflow内嵌的函数现在mnist数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
hist_acc=list()
hist_loss = list()
sess = tf.InteractiveSession()
#Layer1
#从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差。
W_conv1 =tf.Variable(tf.truncated_normal([3, 3, 1, 32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
#调整x的大小
x_image = tf.reshape(x, [-1,28,28,1])
#tf.nn.relu  max(features, 0)
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                       strides=[1, 1, 1, 1], padding='SAME')
#Layer2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#Layer3
W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 64],stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
#Layer4
W_conv4 = tf.Variable(tf.truncated_normal([5, 5, 64, 32],stddev=0.1))
b_conv4 = tf.Variable(tf.constant(0.1,shape=[32]))
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_conv4)
#Layer5
W_conv5 = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1))
b_conv5 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
h_pool3 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
#Layer6-全连接层
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
#对h_pool2数据进行铺平
h_pool2_flat = tf.reshape(h_pool3, [-1, 7*7*64])
#进行relu计算，matmul表示(wx+b)计算
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#Layer7-全连接层
W_fc2 = tf.Variable(tf.truncated_normal([1024,1024],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape=[1024]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#Softmax层
W_fc3 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1,shape=[10]))
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
#在这里通过tf.nn.softmax_cross_entropy_with_logits函数可以对y_conv完成softmax计算，同时计算交叉熵损失函数
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#定义训练目标以及加速优化器
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#初始化变量
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for i in range(4000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0})
    hist_acc.append(test_accuracy)
    loss_val = sess.run(cross_entropy , feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
    hist_loss.append(loss_val)
    print("step %d, testing accuracy %g"%(i, test_accuracy))

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#保存模型
save_path = saver.save(sess, "./model/AlexNet/AlexNet.ckpt")

print("result accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))
plt.plot(hist_acc)
plt.xlabel('step')
plt.ylabel("test accuracy")
plt.title("model accuracy")
plt.show()

plt.plot(hist_loss)
plt.xlabel('step')
plt.ylabel("loss")
plt.title("model loss")
plt.show()