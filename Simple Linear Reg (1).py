#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Simple linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


data=pd.read_csv('Position_Salaries.csv')


# In[5]:


data


# In[ ]:





# In[6]:


# loc,iloc
x=data.iloc[:,1].values
x=x.reshape(-1,1)
x.ndim


# In[7]:


y=data.iloc[:,2].values
y.ndim


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[10]:


x_train


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


reg=LinearRegression()


# In[13]:


reg.fit(x_train,y_train)


# In[14]:


y_pred=reg.predict(x_test)
y_pred
y_pred1=reg.predict(x_train)


# In[15]:


reg.intercept_


# In[16]:


plt.scatter(x,y,color='green')
plt.plot(x,reg.predict(x),color='red')
plt.show()


# In[17]:


reg.coef_


# In[18]:


plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,reg.predict(x_test),color='red')
plt.show()


# In[19]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[20]:


d={'Acutal':y_test,'predict':y_pred,'Error':y_test-y_pred}
data=pd.DataFrame(d)
data


# In[21]:


d={'Acutal':y_train,'predict':y_pred1,'Error':y_train-y_pred1}
data=pd.DataFrame(d)
data


# In[ ]:





# In[ ]:




