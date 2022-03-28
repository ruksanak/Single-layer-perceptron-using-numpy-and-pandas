#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 00:51:13 2022

@author: ruksanakhan
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


from Single_perceptron import Single_layer_perceptron


#Funtion to plot a line
def MSE_plot(MSE_train, MSE_test, title):
    plt.figure
    df = pd.DataFrame({"Train MSE": MSE_train, "Test MSE": MSE_test, "Epoch": np.arange(len(MSE_train))})
    df = df.melt('Epoch', var_name='MSE', value_name='value')
    sns.lineplot(data=df, x='Epoch', y='value', hue='MSE')     
    plt.title("MSE vs Epoch" + title)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()
    plt.close()
    
if __name__ == '__main__':    

    #Dataset charactersticks and description
    iris = load_iris()
    print("Iris data key:")
    print(list(iris.keys()))
    
    print("Iris data description")
    print(iris.DESCR)
    
    #Get x and y matrix    
    x = iris.data
    y = iris.target
    
    #where Sentosa = 0 and all other are 1
    Y = np.where(y>0, 1, 0)
    
    
    #split test and train 
    
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state = 10)
    
    #Make one matrix (combine x and y) , np.c_[np.ones((len(x_train), 1)) - to add bias
    train_matrix = np.column_stack((np.c_[np.ones((len(x_train), 1)), x_train], y_train))
    test_matrix  = np.column_stack((np.c_[np.ones((len(x_test), 1)), x_test], y_test))
    
    
    SLP = Single_layer_perceptron(train_matrix, test_matrix, 5, 0.01, 100) 
    
    
    print("\nBatch Gradient\n")        
    mse_batch_train, mse_batch_test, weight = SLP.batch_gradient()    
    print("Train weights using batch gradient:", weight)
# =============================================================================
#     print("MSE for Batch Gradient (train dataset):", mse_batch_train)
#     print("MSE for Batch Gradient (test dataset):",  mse_batch_test)
# =============================================================================
    MSE_plot(mse_batch_train,  mse_batch_test, ' MSE vs Epoch for Batch Gradient Descent')
    
    

    print("\nStochastic Gradient Descent\n")        
    mse_SGD_train, mse_SGD_test, weight = SLP.SGD()    
    print("Train weights using batch gradient:", weight)
# =============================================================================
#     print("MSE for Stochastic Gradient (train dataset):", mse_SGD_train)
#     print("MSE for Stochastic Gradient (test dataset):",  mse_SGD_test)
# =============================================================================
    MSE_plot(mse_SGD_train,  mse_SGD_test, ' MSE vs Epoch for Stochastic Gradient Descent')    


    print("\nMini batch gradient\n")        
    mse_MB_train, mse_MB_test, weight = SLP.mini_batch(batch_size=12) 
    print("Train weights using batch gradient:", weight)
# =============================================================================
#     print("MSE for Batch Gradient (train dataset):", mse_MB_train)
#     print("MSE for Batch Gradient (test dataset):",  mse_MB_test)
# =============================================================================
    MSE_plot(mse_MB_train,  mse_MB_test, ' MSE vs Epoch for Mini Batch Gradient Descent')    
    
