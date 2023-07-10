# -*- coding: utf-8 -*-
"""LRStochastic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fdkMvkKhIx8YmV-vltkOTZp-kXvv8NHa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Dsata Set for Assignment 1.csv')

"""# **Data Processing**"""

label_dict = {'B': 0, 'M': 1}
df.replace({'diagnosis': label_dict}, inplace = True)

df=df.drop('id',axis=1)

'''
For Learning Task 1 we 
can just remove the feature Engineering task 1 and task 2
'''

##Feature Engineering 1
df = df.fillna(df['diagnosis'].value_counts().index[0])
df.fillna(df.mean(), inplace=True)

result=[]
for i in range (10):
    randX=180+5*i
    train=df.sample(frac=0.67,random_state=randX)
    test=df.drop(train.index)

    X_train, y_train = train.drop('diagnosis', axis = 1), train['diagnosis']
    X_test, y_test = test.drop('diagnosis', axis = 1), test['diagnosis']

    ##Feature Engineering 2
    for col in X_train.columns:
        if (X_train[col].dtypes == 'int64' or X_train[col].dtypes == 'float64') and X_train[col].nunique() > 1:
            X_train[col] = (X_train[col] - X_train[col].mean()) / (X_train[col].std())
    for col in X_test.columns:
        if (X_test[col].dtypes == 'int64' or X_test[col].dtypes == 'float64') and X_test[col].nunique() > 1:
            X_test[col] = (X_test[col] - X_test[col].mean()) / (X_test[col].std())

    """# **Gradient Descent Functions**"""

    def next_batch(x,y,batchsize):
        for i in np.arange(0, x.shape[0], batchSize):
                yield (x[i:i + batchSize], y[i:i + batchSize])

    def sigmoid(x):
        sig=1 / (1 + np.exp(-x))
        return sig

    def log_loss(y, y_dash):
        epsilon=1e-6
        loss = - (y * np.log(y_dash+epsilon)) - ((1 - y) * np.log(1 - y_dash+epsilon))
        return loss

    def cost(y, y_dash):
        m = len(y)
        ccost = 0
        for i in range(m):
            ccost += log_loss(y[i], y_dash[i])
        ccost = ccost/m
        return ccost

    def cost_helper(x,y,w,b):
        m=len(y)
        n=len(w)
        z = []
        for i in range(m):
            s = 0
            for j in range(n):
                s += x[i, j] * w[j]
            z.append(s + b)
        z = np.array(z)
        y_dash = sigmoid(z)
        ccost = cost(y, y_dash)
        return ccost

    def gradients(x,y,w,b):
        m=len(y)
        n=len(w)
        grad_w, grad_b = np.zeros(n), 0
        for i in range(m):
            s = 0
            for j in range(n):
                s += x[i, j] * w[j]
            y_dash_i = sigmoid(s + b)
            for j in range(n):
                grad_w[j] += (y_dash_i  - y[i]) * x[i,j]
            grad_b += y_dash_i  - y[i]
        grad_w, grad_b = grad_w / m, grad_b / m
        return grad_w, grad_b

    def grad_desc(x,y,w,b,alpha,iter,show_cost=False):
        m=len(y)
        n=len(w)
        for i in range(iter):
          grad_w,grad_b=gradients(x,y,w,b)
          w += - alpha * grad_w
          b += - alpha * grad_b
          ccost=cost_helper(x,y,w,b)
        return w, b, ccost

    """# **Model**"""

    w_out = np.zeros(X_train.shape[1])
    b_out = 0.0
    epochs=800
    '''
    batchSize=1 for Stochastic Gradient Descent
    batchSize=40 for Mini-Batch Gradient Descent
    batchSize=len(X_train.to_numpy()) for Batch Gradient Descent
    '''
    batchSize=1      
    alpha=0.1
    cost_history=[]
    ccost=0

    for epoch in range(epochs):
        for (batchX,batchY) in next_batch(X_train.to_numpy(),y_train.to_numpy(),batchSize):
            w_out,b_out,ccost=grad_desc(batchX,
                                        batchY,
                                        w_out,b_out,
                                        alpha,
                                        1)
        cost_history.append(ccost)


    plt.figure(figsize = (9, 6))
    plt.plot(cost_history)
    plt.xlabel("Iteration", fontsize = 14)
    plt.ylabel("Cost", fontsize = 14)
    plt.title("Cost vs Iteration", fontsize = 14)
    plt.tight_layout()
    plt.show()

    """# **Prediction**"""

    y_train_prob = sigmoid(np.matmul(X_train.to_numpy(), w_out) + (b_out * np.ones(X_train.shape[0])))
    y_test_prob = sigmoid(np.matmul(X_test.to_numpy(), w_out) + (b_out * np.ones(X_test.shape[0])))
    tres=0.5
    train_prob=test_prob=0
    y_train=list(y_train)
    y_train_prob=list(y_train_prob)
    y_test=list(y_test)
    y_test_prob=list(y_test_prob)
    for i in range(len(y_train)):
      if (y_train[i]==1) and (y_train_prob[i]>=tres):
        train_prob+=1
      elif (y_train[i]==0) and (y_train_prob[i]<tres):
        train_prob+=1
    for i in range(len(y_test)):
      if (y_test[i]==1) and (y_test_prob[i]>=tres):
        test_prob+=1
      elif (y_test[i]==0) and (y_test_prob[i]<tres):
        test_prob+=1
    print("Training Accuracy ", (train_prob/len(y_train))*100)
    print("Testing Accuracy ", (test_prob/len(y_test))*100)
    result.append((test_prob/len(y_test))*100)
print("Mean ",np.mean(result))
print("Variance ",np.var(result))