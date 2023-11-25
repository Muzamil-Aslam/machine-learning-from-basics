#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#polynomial regression


# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('Position_Salaries.csv')


# In[4]:


data


# In[5]:


x=data.iloc[:,1].values


# In[6]:


x=x.reshape(-1,1)


# In[7]:


x


# In[8]:


y=data.iloc[:,2].values


# In[9]:


y


# In[10]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[11]:


reg=LinearRegression()
reg.fit(x,y)


# In[19]:


reg_poly=PolynomialFeatures(degree=3)


# In[20]:


reg_poly


# In[21]:


x_poly=reg_poly.fit_transform(x)


# In[22]:


x_poly


# In[23]:


reg_poly.fit(x_poly,y)


# In[ ]:





# In[ ]:





# In[24]:


reg=LinearRegression()
reg.fit(x_poly,y)


# In[27]:


# Visualising the Linear Regression results
plt.scatter(x, y, color='blue')
 
plt.plot(x, reg.predict(x), color='red')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


plt.scatter(x,y,color="b")
plt.plot(x,reg.predict(x),color="g")
plt.show()


# In[ ]:


plt.scatter(x,y,color="b")
plt.plot(x,reg1.predict(x_poly),color="g")
plt.show()


# In[ ]:


from sklearn.metrics import r2_score
a=r2_score(x,reg1.predict(x_poly))


# In[ ]:


a


# In[ ]:


b=r2_score(x,reg.predict(x))


# In[ ]:


b


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




