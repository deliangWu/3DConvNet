import numpy as np
import cv2
from functools import reduce
import tensorflow as tf
import time
import ut_interaction
import videoPreProcess


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.1)
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

def max_pool3d_2x1x1(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 1, 1, 1],
                            strides=[1, 2, 1, 1, 1], padding='SAME')

def max_pool3d_4x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 2, 2, 1],
                            strides=[1, 4, 2, 2, 1], padding='SAME')

class FeatureDescriptor:
    @staticmethod
    def c3d(x,frmSize,drop_var):
        # define the first convlutional layer
        W_conv1 = weight_variable([3,3,3,frmSize[2],64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool3d_1x2x2(h_conv1)    
    
        # define the second convlutional layer
        W_conv2 = weight_variable([3,3,3,64,128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool3d_2x2x2(h_conv2)    
    
        # define the 3rd convlutional layer
        W_conv3a = weight_variable([3,3,3,128,256])
        b_conv3a = bias_variable([256])
        h_conv3a = tf.nn.relu(conv3d(h_pool2, W_conv3a) + b_conv3a)
        W_conv3b = weight_variable([3,3,3,256,256])
        b_conv3b = bias_variable([256])
        h_conv3b = tf.nn.relu(conv3d(h_conv3a, W_conv3b) + b_conv3b)
        h_pool3 = max_pool3d_2x2x2(h_conv3b)    
    
        # define the 4rd convlutional layer
        W_conv4a = weight_variable([3,3,3,256,512])
        b_conv4a = bias_variable([512])
        h_conv4a = tf.nn.relu(conv3d(h_pool3, W_conv4a) + b_conv4a)
        W_conv4b = weight_variable([3,3,3,512,512])
        b_conv4b = bias_variable([512])
        h_conv4b = tf.nn.relu(conv3d(h_conv4a, W_conv4b) + b_conv4b)
        h_pool4 = max_pool3d_2x2x2(h_conv4b)    
    
        # define the 5rd convlutional layer
        W_conv5a = weight_variable([3,3,3,512,512])
        b_conv5a = bias_variable([512])
        h_conv5a = tf.nn.relu(conv3d(h_pool4, W_conv5a) + b_conv5a)
        W_conv5b = weight_variable([3,3,3,512,512])
        b_conv5b = bias_variable([512])
        h_conv5b = tf.nn.relu(conv3d(h_conv5a, W_conv5b) + b_conv5b)
        h_pool5 = max_pool3d_2x1x1(h_conv5b)    
    
        # define the full connected layer
        W_fc6 = weight_variable([int(frmSize[0]/16 * frmSize[1]/16) * 512, 4096])
        b_fc6 = bias_variable([4096])
        h_pool5_flat = tf.reshape(h_pool5, [-1, int(frmSize[0]/16 * frmSize[1]/16) * 512])
        h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)  
        h_fc6_drop = tf.nn.dropout(h_fc6, drop_var) 
        h_fc6_l2norm = tf.nn.l2_normalize(h_fc6_drop,dim=1)
    
        # define the full connected layer fc7
        W_fc7 = weight_variable([4096, 4096])
        b_fc7 = bias_variable([4096])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6_l2norm, W_fc7) + b_fc7)
        h_fc7_drop = tf.nn.dropout(h_fc7, drop_var)
        
        return h_fc7_drop

class Classifier:
    @staticmethod
    def softmax(features,fDim,numOfClasses):
        # softmax
        W_sm = weight_variable([fDim, numOfClasses])
        b_sm = bias_variable([numOfClasses])
        y_conv = tf.matmul(features, W_sm) + b_sm 
        
        return y_conv
        
def main(_):
    numOfClasses = 6 
    frmSize = (112,128,3)
    # define the dataset
    ut_set1 = ut_interaction.ut_interaction_set1(frmSize)
    print('ok')

    # build the 3D ConvNet
    # define the input and output variables
    x = tf.placeholder(tf.float32, (None,16) + frmSize)
    y_ = tf.placeholder(tf.float32, [None, numOfClasses])
    keep_prob = tf.placeholder(tf.float32)
    features = FeatureDescriptor.c3d(x,frmSize,keep_prob)
    y_conv = Classifier.softmax(features,4096,numOfClasses)
    print('ok')

    # Train and evaluate the model
    sess = tf.Session()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with sess.as_default():
        #for d in ['/gpu:0', '/gpu:1']:
        for d in ['/cpu:0']:
            with tf.device(d):
                # train the 3D ConvNet
                for seq in range(1,11):
                    sess.run(tf.global_variables_initializer())
                    #saver.restore(sess,'../models/model._0117.ckpt')
                    print('start to train and test on sequence ',seq)
                    ut_set1.splitTrainingTesting(seq)
                    test_x,test_y = ut_set1.loadTesting()[0:2]

                    for i in range(400):
                        train_x,train_y = ut_set1.loadTraining(1) 
                        if i%20 == 0:
                            train_accuracy = accuracy.eval(feed_dict={
                                x:train_x, y_: train_y, keep_prob: 1.0})
                            print("step %d, training accuracy %g"%(i, train_accuracy))
                            print("test accuracy %g"%accuracy.eval(feed_dict={
                                x: test_x, y_: test_y, keep_prob: 1.0}))    
                        train_step.run(feed_dict={x: train_x, y_: train_y, keep_prob: 0.5})
                    # test the trained network performance
                    #saver.save(sess,'./model_0117.ckpt')
                    #saver.restore(sess,'./model.ckpt')
                    print ('Begin to Test the 3D ConvNet')
                    print("test accuracy %g"%accuracy.eval(feed_dict={
                        x: test_x, y_: test_y, keep_prob: 1.0}))    
                    y_val = y_conv.eval(feed_dict = {x:test_x,keep_prob:1.0})
                    top_n_perf(y_val,test_y,1)

if __name__ == "__main__":
    tf.app.run(main=main)
