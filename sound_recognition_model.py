# -*- coding: utf-8 -*-
"""
Created on Wed Jul 8 16:44:14 2019

@author: niuzhengnan
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import scipy
from PIL import Image
from scipy import ndimage
from data_preprocess import *

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    """
    X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")   
    return X, Y

def initialize_parameters(n_l1, n_l2, n_l3, length_of_sample):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_l1, length_of_sample]
                        b1 : [n_l1, 1]
                        W2 : [n_l2, n_l1]
                        b2 : [n_l2, 1]
                        W3 : [n_l3, n_l2]
                        b3 : [n_l3, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """    
        
    W1 = tf.compat.v1.get_variable("W1", [n_l1, length_of_sample], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.compat.v1.get_variable("b1", [n_l1, 1], initializer = tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [n_l2, n_l1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [n_l2, 1], initializer = tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [n_l3, n_l2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [n_l3, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']   
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                               # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                               # Z3 = np.dot(W3,Z2) + b3    
    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate =0.001,
          num_epochs = 10, minibatch_size = 32, print_cost = True, n_l1=25, n_l2=12, n_l3=7):
    """learning_rate = 0.0001,
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size, number of training examples)
    Y_train -- test set, of shape (output size, number of training examples)
    X_test -- training set, of shape (input size, number of training examples)
    Y_test -- test set, of shape (output size, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    n_l3 depends on how many kinds of sound we want to recognize.
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
#    tf.set_random_seed(1)                             # to keep consistent results
#    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_l1, n_l2, n_l3, n_x)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 =  forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer =tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.compat.v1.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
#            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #test = sess.run(accuracy,feed_dict = {X: X_train, Y: Y_train})
        
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters, accuracy
        
if __name__ == "__main__":    
    
    # label 0 : Bike  110/18
#    for i in range(1,110):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Bike\Bike_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)#
#        add_into_Trainset(res, 0, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(111,128):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Bike\Bike_%d.wav'%i)
 #       res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 0, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy") 
# label 1 : Car 125/20    
#    for i in range(1,125):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 1, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(125,145):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 1, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
## label 2: Emergency vehicle  (107/13)
#    
##    for i in range(1,108):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Emergencyvehicle\Emergencyvehicle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 2, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(108,121):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Emergencyvehicle\Emergencyvehicle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 2, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
#    
# # Label 3:Horn 141/30
#    
#    for i in range(1,142):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Horn\Horn_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 3, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(142,172):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Horn\Horn_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 3, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
        
        
 # Label 4 : Motorcycle 107/13
    
#    for i in range(1,108):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Motorcycle\Motorcycle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 4, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(142,121):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Motorcycle\Motorcycle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 4, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")    
        
 # Label 5 : Noise 147/30
    
#    for i in range(1,148):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Noise\\Noise_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 5, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(148,178):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Noise\\Noise_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 5, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")     
        
 # Label 6 : rail 110/13
    
#    for i in range(1,111):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Rail\\rail_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 6, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(111,124):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Rail\\rail_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 6, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")  
    
    path_x_Train ='C:/Users/niuzhengnan/Desktop/sr/x_train.npy'
    x_train = np.load(path_x_Train)
    x_train = x_train.T
    path_y_Train = 'C:/Users/niuzhengnan/Desktop/sr/y_train.npy'
    y_train_orig = np.load(path_y_Train)
    y_train = convert_to_one_hot(y_train_orig, 7)
    path_x_Test ='C:/Users/niuzhengnan/Desktop/sr/x_test.npy'
    path_y_Test ='C:/Users/niuzhengnan/Desktop/sr/y_test.npy'
    x_test = np.load(path_x_Test)
    x_test = x_test.T
    y_test_orig = np.load(path_y_Test)
    y_test = convert_to_one_hot(y_test_orig, 7)
    parameters,accuracy = model(x_train, y_train, x_test, y_test,n_l3=7)