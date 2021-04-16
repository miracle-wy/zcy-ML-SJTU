# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 每个批次的大小
batch_size = 100
his_acc = list()
his_loss = list()
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 输入图像，定义占位符
y = tf.placeholder(tf.float32, [None, 10])  # 输入标签

# 创建一个简单的神经网络 784个像素点对应784个数  因此输入层是784个神经元 输出层是10个神经元 不含隐层
# 最后准确率在92%左右 BP神经网络
W = tf.Variable(tf.zeros([784, 10]))  # 生成784行 10列的全0矩阵 变量定义
b = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)  # tf.matmul是矩阵相乘softmax归一化，得到分类概论

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))  # 所有值求差值平方的均值
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)  # 0.2为学习率，目标最小化loss

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在布尔型列表中
# argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 逐个判断，返回也是一个同维度矩阵，元素为布尔
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast为张量类型转换
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):  # 100个epoch 把所有的图片训练100次
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,loss_val = sess.run([train_step,loss], feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000]})
        his_acc.append(acc)
        his_loss.append(loss_val)
        print("step " + str(epoch) + ",Testing Accuracy " + str(acc))
    print("result accuracy %g" % accuracy.eval(feed_dict={
     x: mnist.test.images, y: mnist.test.labels}))
    save_path = saver.save(sess, "./model/BP/BP.ckpt")
plt.plot(his_acc)
plt.xlabel('step')
plt.ylabel("test accuracy")
plt.title("model accuracy")
plt.show()

plt.plot(his_loss)
plt.xlabel('step')
plt.ylabel("loss")
plt.title("model loss")
plt.show()