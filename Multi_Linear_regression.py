#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('50_Startups.csv')
data


# In[3]:


x=data.drop('Profit',axis=1).values
x=x[:,0:3]
x


# In[4]:


y=data['Profit'].values
y


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[7]:


lr.fit(x_train,y_train)


# In[8]:


c=lr.intercept_
c


# In[9]:


m=lr.coef_
m


# In[10]:


y_pred=lr.predict(x_train)


# In[11]:


y_pred


# In[12]:


plt.figure(figsize=(8,8))
plt.scatter(y_train,y_pred)
plt.xlabel('Actual values')
plt.ylabel('predicted values')
plt.show()


# In[13]:


from sklearn.metrics import r2_score


# In[14]:


r2_score(y_train,y_pred)*100


# In[15]:


y_pred_test=lr.predict(x_test)


# In[16]:


plt.figure(figsize=(5,3))
plt.scatter(y_test,y_pred_test)
plt.xlabel('Actual values')
plt.ylabel('predicted values')
plt.show()


# In[20]:


r2_score(y_test,y_pred_test)*100


# In[21]:


from sklearn.metrics import mean_squared_error


# In[22]:


mean_squared_error(y_test,y_pred_test)*100


# In[ ]:




