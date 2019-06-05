###############################################
#Author:Amith A Shetty
##############################################
#Accuracy:89.17%
###########################################
#This is a neural network implemantation from scratch.
#Data set is MNIST and there are training set and test set
##################################################

import numpy as np
import pandas as pd
import pickle as pk

####Activation function used is sigmoid function############
def sigmoidFunc(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))




#testing data set
data = np.array(pd.read_csv('mnist_test.csv',sep=',',header=None))

#storing the weights
f =  open("weights","rb")
datastore = pk.load(f)
syn0  = datastore[0]
syn1  = datastore[1]
f.close()



#function to find max vlue in output
def check(arr):
    return np.argmax(arr)

#assigning data set values
x= data[:,1:]
y = data[:,0]
sum = 0

#calculating percentage  
l0 = x
l1 = sigmoidFunc(np.dot(l0,syn0))
l2 = sigmoidFunc(np.dot(l1,syn1))


#checking for answers
for i in range(len(l2)):
    ans =  check(l2[i])
    if ans == y[i]:
        sum += 1


#calculating percentage 
perc = (sum/len(l2))*100

print("The percentage of accuracy is:%r" %(perc))
