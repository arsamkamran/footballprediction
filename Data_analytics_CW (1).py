#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#from sklearn.datasets import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import neighbors
dataset = pd.read_csv("/home/arsam/Downloads/seasons_2005_2019_new.csv")
dataset.head()


# In[3]:


plt.scatter(dataset.iloc[:,3:4],dataset.iloc[:, 9:10])
plt.show()


# In[4]:


#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
team_encoded=le.fit_transform(dataset.loc[:,"HomeTeam"])
print(team_encoded)


# In[5]:


x_prime = dataset.iloc[:,[9,10,11,12,13,14,15,16,17,18,19,20]].values
y = dataset.iloc[:,5:6].values


# In[6]:


x_prime


# In[7]:


X = preprocessing.scale(x_prime)


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=4)


# In[9]:


clf = neighbors.KNeighborsClassifier(n_neighbors=25)
clf.fit(x_train,y_train)
print(clf)
y_expect = y_test
y_pred = clf.predict(x_test)


# In[10]:


print(metrics.classification_report(y_expect,y_pred))


# In[11]:


from sklearn.metrics import accuracy_score
print('Accuracy Score:',accuracy_score(y_test,y_pred)*100,'%')


# In[12]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[13]:


dataset.describe()


# In[59]:


# Linear regression

reg = linear_model.LinearRegression()
reg_x_prime = dataset.iloc[:,[9,10,11,12,13,14,15,16,17,18,19,20]].values
reg_y = dataset.iloc[:,3:4].values
reg_x_train,reg_x_test,reg_y_train,reg_y_test = train_test_split(X,reg_y,test_size=0.20, random_state=4)
reg.fit(reg_x_train,reg_y_train)


# In[60]:


reg_y_expect = reg_y_test
reg_y_pred = reg.predict(reg_x_test)


# In[61]:


reg_y_expect


# In[62]:


rounded_pred = np.around(reg_y_pred)
rounded_pred


# In[63]:


import numpy as np
np.mean((rounded_pred-reg_y_expect)**2)


# In[64]:


print('Accuracy Score:',accuracy_score(reg_y_expect,rounded_pred)*100,'%')


# In[67]:


from sklearn.naive_bayes import MultinomialNB


# In[69]:


MultiNB = MultinomialNB()
MultiNB.fit(np.absolute(x_train),y_train)
print(MultiNB)
nb_y_pred = MultiNB.predict(x_test)
print('Accuracy Score:',accuracy_score(y_test,nb_y_pred)*100,'%')


# In[ ]:
