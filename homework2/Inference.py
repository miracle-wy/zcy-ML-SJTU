from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def imageprepare():
    im = Image.open('3.jpg')
    plt.imshow(im)
    plt.show()
    data = list(im.getdata())
    return data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def predict(imvalue):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
# reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# ********************** conv1 *********************************
# transfer a 5*5*1 imagine into 32 sequence
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

# ********************** conv2 *********************************
# transfer a 5*5*32 imagine into 64 sequence
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

# ********************* func1 layer *********************************
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
# reshape the image from 7,7,64 into a flat (7*7*64)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# output size 1*1024
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ********************* func2 layer *********************************
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "./model/LeNet/LeNet.ckpt")
        # print ("Model restored.")
        result = tf.argmax(prediction, 1)
        return result.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)



if __name__ == "__main__":
     img = imageprepare()
     print("This number is" + str(predict(img)))
