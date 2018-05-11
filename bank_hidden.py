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
print 'Training size: ', len(X_train[1])


#########

#hyperparameters
hid = 20
alpha = 0.001

def sigmoid(z):
    eq = 1/(1+(np.exp(-z)))
    return eq

w1 = np.random.randn(hid,4)
b1 = np.zeros(shape = (hid,1))
w2 = np.random.randn(1,hid)
b2 = np.zeros(shape = (1,1))
m = y_train.shape[1]


for i in range(0,10000):
    Z1 = np.dot(w1,X_train) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1) + b2
    A2 = sigmoid(Z2)

    logprobs = np.multiply(np.log(A2), y_train) + np.multiply((1 - y_train), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    
    dZ2 = (A2 - y_train)
    dw2 = (1.0/m) * np.dot(dZ2,A1.T)
    db2 = (1.0/m) * np.sum(dZ2, axis = 1, keepdims=True)
    
    #this is tanh deriv
    dA1 = (np.dot(w2.T,dZ2))
    dZ1 = dA1*(1 - np.tanh(Z1)**2)
    
    dw1 = (1.0/m) * np.dot(dZ1,X_train.T)
    db1 = (1.0/m) * np.sum(dZ1, axis = 1, keepdims=True)
    
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    
    if i % 1000 == 0:
    
        print ("Cost after iteration %i: %f" %(i, cost))
print ("Cost after iteration %i: %f" %(i, cost))



#prediction
Z1 = np.dot(w1,X_test.T) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(w2,A1) + b2
A2 = sigmoid(Z2)
predictions = np.round(A2)


#print predictions.T.shape
#print y_test.T.shape


print ('Test Accuracy: %.2f' % float((np.dot(y_test.T,predictions.T) +
                             np.dot(1-y_test.T,1-predictions.T))/float(y_test.size)*100) + '%')


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


