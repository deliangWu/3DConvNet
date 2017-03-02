import numpy as np
import cv2
from functools import reduce
import tensorflow as tf
import time
import ut_interaction
import videoPreProcess

def videoNorm(videoIn):
    vmax = np.amax(videoIn)
    vmin = np.amin(videoIn)
    vo = (videoIn - vmin)/(vmax-vmin) * 255
    vo = tf.cast(vo,dtype=tf.uint8)
    vout = vo.eval()
    return vout


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1,1], padding='SAME')

def max_pool3d_1x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                             strides=[1, 1, 2, 2, 1], padding='SAME')

def max_pool3d_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool3d_4x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 2, 2, 1],
                             strides=[1, 4, 2, 2, 1], padding='SAME')
    
def top_n_perf(y_val,y_,n=3):    
    top3y = np.argsort(y_val,axis=1)
    topy = np.argmax(y_val,axis=1)
    cnt_correct = 0
    y_gt = np.argmax(y_,axis=1)
    for i in range(y_.shape[0]):
        if y_gt[i] in top3y[i,-n::]:
            result = 'correct'
            cnt_correct += 1
        else:
            result = 'error'
    top3accu = np.float(cnt_correct) / y_.shape[0]
    print('the final top',n,' accuracy is ',top3accu)
def main(_):
    numOfClasses = 6 
    frmSize = (112,128)
    # define the dataset
    ut_set1 = ut_interaction.ut_interaction_set1(frmSize)
    print('ok')
    
    # build the 3D ConvNet
    # define the input and output variables
    x = tf.placeholder(tf.float32, (None,16) + frmSize + (3,))
    y_ = tf.placeholder(tf.float32, [None, numOfClasses])
    print('ok')
    
    # define the first convlutional layer
    W_conv1 = weight_variable([3,3,3,3,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool3d_1x2x2(h_conv1)    
    
    # define the second convlutional layer
    W_conv2 = weight_variable([3,3,3,32,128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool3d_2x2x2(h_conv2)    
    
    # define the 3rd convlutional layer
    W_conv3 = weight_variable([3,3,3,128,256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool3d_2x2x2(h_conv3)    
    
    # define the 3rd convlutional layer
    W_conv4 = weight_variable([3,3,3,256,512])
    b_conv4 = bias_variable([512])
    h_conv4 = tf.nn.relu(conv3d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool3d_4x2x2(h_conv4)    
    
    # define the full connected layer
    W_fc1 = weight_variable([int(frmSize[0]/16 * frmSize[1]/16) * 512, 4096])
    b_fc1 = bias_variable([4096])
    
    h_pool2_flat = tf.reshape(h_pool4, [-1, int(frmSize[0]/16 * frmSize[1]/16) * 512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    

    # dropout layer
    W_fc2 = weight_variable([4096, numOfClasses])
    b_fc2 = bias_variable([numOfClasses])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2    
    
    # Train and evaluate the model
    sess = tf.Session()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Create a saver
    saver = tf.train.Saver()
    
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        with tf.device("/cpu:0"):
            # train the 3D ConvNet
            print('ok3')
            ut_set1.splitTrainingTesting(3)
            print('ok4')
            for i in range(100):
                train_x,train_y = ut_set1.loadTraining() 
                print("i = ", i)
                if i%5 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                                     x:train_x, y_: train_y, keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: train_x, y_: train_y, keep_prob: 0.5})
            # test the trained network performance
            #saver.save(sess,'./model_0117.ckpt')
            #saver.restore(sess,'./model.ckpt')
            print ('Begin to Test the 3D ConvNet')
            test_x,test_y = ut_set1.loadTesting()
            print("test accuracy %g"%accuracy.eval(feed_dict={
                   x: test_x, y_: test_y, keep_prob: 1.0}))    
            y_val = y_conv.eval(feed_dict = {x:test_x,keep_prob:1.0})
            top_n_perf(y_val,test_y,1)
    
if __name__ == "__main__":
    tf.app.run(main=main)