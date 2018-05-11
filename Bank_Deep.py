'''
3 layer neural net with l2 regularization applied to weights
'''


# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *



#########DATA##########

df=pd.read_csv('data_banknote_authentication.txt', sep=',',header=None)
data = df.values

ylist = []
Xlist = []
for i in range(len(data)):
    ylist.append(data[i][4:])
    Xlist.append(data[i][0:4])
    
X = np.asarray(Xlist)
y = np.asarray(ylist)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
features = X_train.shape[1]
#print y_train.shape
#print X_train


####################################

###training/Test#####
from sklearn.model_selection import train_test_split

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#show dimensions

X_train = X_train.T
y_train = np.reshape(y_train,(len(y_train),1))
y_train = y_train.T


#print 'X_train shape:', X_train.shape
#print 'y_train shape:', y_train.shape
print 'Training set size: ', len(X_train[1])



#########

#hyperparameters
alpha = 0.0025
lambd = 0.1

def sigmoid(z):
    eq = 1/(1+(np.exp(-z)))
    return eq

w1 = np.random.rand(10,4) 
b1 = np.zeros(shape = (10,1))
w2 = np.random.rand(5,10)
b2 = np.zeros(shape = (5,1))
w3 = np.random.randn(1,5)
b3 = np.zeros(shape = (1,1))

m = y_train.shape[1]



for i in range(0,10000):
    #forward prop
    Z1 = np.dot(w1,X_train) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(w3, A2) + b3
    A3 = sigmoid(Z3)
    #print 'a3', A3.shape

    
    
    #cost
    logprobs = np.multiply(np.log(A3), y_train) + np.multiply((1 - y_train), np.log(1 - A3))
    #logprobs = (1,919)
    cost = - np.sum(logprobs) / m

    weight_sum = np.sum(w1) + np.sum(w2) + np.sum(w3)
    #weight sum a number 

    L2_regularization_cost = lambd/(2*m) * weight_sum
    #L2 a number

    cost = cost + L2_regularization_cost


    
    #layer 3
    dZ3 = (A3 - y_train)
    #print 'dz3', dZ3.shape
    dw3 = (1.0/m) * np.dot(dZ3,A2.T) + lambd/m * w3
    db3 = (1.0/m) * np.sum(dZ3, axis = 1, keepdims=True)
    #print 'dw3', dw3.shape
    
    #layer 2
    dA2 = (np.dot(w3.T,dZ3))
    #print 'da2', dA2.shape
    dZ2 = dA2*(1 - np.tanh(Z2)**2)
    #print 'dz2', dZ2.shape
    
    dw2 = (1.0/m) * np.dot(dZ2,A1.T) + lambd/m * w2
    db2 = (1.0/m) * np.sum(dZ2, axis = 1, keepdims=True)
    #print 'dw2', dw2.shape

    #layer 1
    dA1 = (np.dot(w2.T,dZ2))
    dZ1 = dA1*(1 - np.tanh(Z1)**2)
    
    
    dw1 = (1.0/m) * np.dot(dZ1,X_train.T) + lambd/m * w1
    db1 = (1.0/m) * np.sum(dZ1, axis = 1, keepdims=True)
    #print 'dw1', dw1.shape
    
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    
    if i % 1000 == 0:
    
        print ("Cost after iteration %i: %f" %(i, cost))
print ("Cost after iteration %i: %f" %(i, cost))



#prediction
Z1 = np.dot(w1,X_test.T) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(w2,A1) + b2
A2 = np.tanh(Z2)
Z3 = np.dot(w3,A2) + b3
A3 = sigmoid(Z3)

predictions = np.round(A3)

#print predictions.T.shape
#print y_test.T.shape

float((np.dot(y_test.T,predictions.T) + np.dot(1-y_test.T,1-predictions.T))/float(y_test.size)*100) 


print ('Test Accuracy: %.2f' % float((np.dot(y_test.T,predictions.T) + 
                    np.dot(1-y_test.T,1-predictions.T))/float(y_test.size)*100) + '%' )




#check how many correct

correct = 0
wrong = 0
for i in range(len(predictions[0])):
    if (predictions[0][i] == y_test[i]) == True:
        correct += 1
    else:
        wrong += 1

print ('Correct: %d' %correct)
print ('Incorrect: %d' %wrong)


'''best without regularizatoin
Test Accuracy: 95.36%
Correct: 432
Incorrect: 21
'''

