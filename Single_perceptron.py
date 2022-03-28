#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 00:23:23 2022

@author: ruksanakhan
"""

import numpy as np
from sklearn.datasets import load_iris 
iris = load_iris()
from sklearn.metrics import mean_squared_error


#number_wts - number of independent_variable+1(bias)
#t0, t1 - learning schedule hyperparameter 

class Single_layer_perceptron:
    #Cost function 

    def __init__(self, train_matrix, test_matrix, number_wts, learning_rate, Epoch):
        
        self.train_matrix = train_matrix
        self.test_matrix  = test_matrix
        self.learning_rate= learning_rate
        self.Epoch        = Epoch
        self.number_wts   = number_wts

    def cost_MSE(self, X, Y, weights):       
        if len(X) != 0:
            return mean_squared_error(Y, np.where(X.dot(weights)>.5, 1, 0))
        
    def mini_batch(self, batch_size=12):
        
        MSE_train_mb = np.zeros(self.Epoch)
        MSE_test_mb  = np.zeros(self.Epoch)
        weight       = np.random.rand(self.number_wts,1) 
            
        n_mini_batches = self.train_matrix.shape[0]//batch_size
            
        x_test = self.test_matrix[:, :-1]
        y_test = self.test_matrix[:, -1].reshape((-1,1))
        
        #shuffle the matrix randomly 
        np.random.shuffle(self.train_matrix)
              
        for epoch in range(self.Epoch):
            for i in range(n_mini_batches):
                mini_batch = self.train_matrix[i * batch_size: (i+1) * batch_size, :]
                x_mini     = mini_batch[:, :-1]
                y_mini     = mini_batch[:, -1].reshape((-1,1))
                 
                gradients = 2/x_mini.shape[0] * (x_mini.T.dot(np.where(x_mini.dot(weight)> .5,1,0) - y_mini))
                
                weight = weight - self.learning_rate * gradients
                 
                MSE_train_mb[epoch] = self.cost_MSE(x_mini, y_mini, weight)
                MSE_test_mb[epoch]  = self.cost_MSE(x_test, y_test, weight)

        return MSE_train_mb, MSE_test_mb, weight         
                
    #Batch gradient 
    def batch_gradient(self):
        
        #Define MSE_train as all zeros
        MSE_train_batch = np.zeros(self.Epoch)
        MSE_test_batch  = np.zeros(self.Epoch)
        
        #Initiate the weight (0,1)
        x_train = self.train_matrix[:, :-1]
        y_train = self.train_matrix[:, -1].reshape((-1,1))
        
        
        x_test = self.test_matrix[:, :-1]
        y_test = self.test_matrix[:, -1].reshape((-1,1))
        
        #number_wts is number of independent_variable+1(bias)
        weight    = np.random.rand(self.number_wts,1)
        
        for epoch in range(self.Epoch):        
    
            gradients = 2/x_train.shape[0] * (x_train.T.dot(np.where(x_train.dot(weight) >.5, 1, 0) - y_train))
              
            weight = weight - self.learning_rate * gradients
            MSE_train_batch[epoch] = self.cost_MSE(x_train, y_train, weight)
            MSE_test_batch[epoch]  = self.cost_MSE(x_test, y_test, weight)
            
            
        return MSE_train_batch, MSE_test_batch, weight
    
        
    #Stochastic Gradient descent
    def SGD(self):
           #Define MSE_train as all zeros
        MSE_train_sgd = np.zeros(self.Epoch)
        MSE_test_sgd  = np.zeros(self.Epoch)
        
        #Initiate the weight (0,1)
        x_train = self.train_matrix[:, :-1]
        y_train = self.train_matrix[:, -1].reshape((-1,1))
        
        
        x_test = self.test_matrix[:, :-1]
        y_test = self.test_matrix[:, -1].reshape((-1,1))
        
        weight    = np.random.rand(self.number_wts,1)
        
        
        for epoch in range(self.Epoch):
            
            for i in range(len(x_train)):
                
          
                     rand_ind = np.random.randint(len(x_train))
    
                     xi  =x_train[rand_ind:rand_ind+1]
                     yi  = y_train[rand_ind:rand_ind+1]
    
                     gradients = 2*xi.T.dot(np.where(xi.dot(weight)>.5, 1, 0) - yi)
                     weight = weight - self.learning_rate * gradients 
                     
                     MSE_train_sgd[epoch] = self.cost_MSE(x_train, y_train, weight)
                     MSE_test_sgd[epoch]  = self.cost_MSE(x_test, y_test, weight)
    
        return MSE_train_sgd, MSE_test_sgd, weight
    
       
          
    

        
        
