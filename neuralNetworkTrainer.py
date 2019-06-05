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
import math


####Activation function used is sigmoid function############
def sigmoidFunc(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
#mnist train dataset in csv format
data = np.array(pd.read_csv('mnist_train.csv',sep=',',header=None))

#This converts decimal mnst label data to required format("1" value incicating its index is that number )
def dectocode(deci):
    data = np.array([0,0,0,0,0,0,0,0,0,0])
    data[deci] = 1
    return data


#learning rate
alpha = 0.00001




#training data
x = data[:,1:]
#labels
y = []
for i in data[:,0]:
    y.append(dectocode(i))
y=np.array(y)




#seed for debugging to get same random data
# np.random.seed(1)
################################################3
# randomly initialize our weights with mean 0
# syn0 = 2* np.random.random((784,10)) - 1
# syn1 = 2* np.random.random((10,10)) - 1
####################OR############################3
#we can initialize weights to zero or use saved vale
#loading saved weights
f =  open("weights","rb")
datastore = pk.load(f)
syn0  = datastore[0]
syn1  = datastore[1]
f.close()


#try block to stop training when key pressed (ctrl+c) 
try:
    for j in range(10000):

        # Feed forward layers
        l0 = x
        l1 = sigmoidFunc(np.dot(l0,syn0))
        l2 = sigmoidFunc(np.dot(l1,syn1))

        # error
        l2_error = y - l2
        
        #updating ourself with error 
        if (j % 100) == 0:
            print("%r Error: %r" %(j,str(np.mean(np.abs(l2_error)))))
            
        # direction to change
        l2_delta = l2_error*sigmoidFunc(l2,deriv=True)

        # finding error
        l1_error = l2_delta.dot(syn1.T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * sigmoidFunc(l1,deriv=True)

        syn1 += alpha*l1.T.dot(l2_delta)
        syn0 += alpha*l0.T.dot(l1_delta)

except KeyboardInterrupt:
    pass

#testing data set
data = np.array(pd.read_csv('mnist_test.csv',sep=',',header=None))


#function to find max vlue in output
def check(arr):
    return np.argmax(arr)

#assigning data set values
x= data[:,1:]
y = data[:,0]
sum = 0

#running the network
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


#storing the weights
datastore = []
datastore.append(syn0)
datastore.append(syn1)
f =  open("weightsData",'wb')
pk.dump(datastore,f)
f.close()


