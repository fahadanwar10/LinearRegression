#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression on E-commerce Customer Data
# 
# We will try to fit a linear regression model on E-commerce Data and try to predict the Yearly amount spent by a customer.

# In[1]:


# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv('Ecomm-Customers.csv')


# In[3]:


customers.info()


#  info() tells us that there are 8 columns and 500 rows . Let us peak into the data using head()

# In[4]:


customers.head()


# Using pairplot to see if there is some sort of correlation among columns with respect to yearly amount spent.

# In[5]:


sns.pairplot(customers)


# From the pair plots, we can see that data distribution is quite normal, and that there is a clear correlation between length of membership and yearly amount spent.<br>
# Let us find out more using heatmap

# In[6]:


sns.heatmap(customers.corr(), linewidth=0.5, annot=True)


# The above heatmap confirms the correlation between 'length of membership' and 'Yearly amount spent'. We can also see that there is good degree of correlation between 'Yearly amount spent' and the column 'Time on app'. Also lesser degree of correlation with 'Avg. Session length'

# In[7]:


x = customers[['Time on App', 'Length of Membership']]
y = customers['Yearly Amount Spent']


# For the time being let's skip 'Avg. Session Length' column since it has lesser correlation. We shall include it later and see if it yields considerably better results.

# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)


# Splitting dataset into train and test , giving 30% as test data and 70% as train data

# In[9]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[10]:


print("Coeffs are Time on App : {0} , Length of Membership: {1}".format(lm.coef_[0], lm.coef_[1]))


# In[11]:


result = lm.predict(x_test)


# In[12]:


plt.scatter(y_test, result)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")


# In[13]:


print('R2 score : ',metrics.r2_score(y_test, result))
print('Variance: ',metrics.explained_variance_score(y_test,result))
print('MSE: ', metrics.mean_squared_error(y_test,result))


# The predicted values and actual values seem to be agreeing with each other and the R2 score is also ~ 0.88, which is seems good enough. But the MSE seems to be higher (not sure).
# However, Let us add the column 'Avg. Session length' this time and check results to see if there's any improvement

# In[14]:


x = customers[['Time on App', 'Length of Membership','Avg. Session Length']]


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)


# Splitting dataset into train and test , giving 30% as test data and 70% as train data

# In[16]:


lm.fit(x_train, y_train)


# In[17]:


print("Coeffs are Time on App : {0} , Length of Membership: {1} , Avg. Session Length: {2}".format(lm.coef_[0], lm.coef_[1], lm.coef_[2]))


# In[18]:


result = lm.predict(x_test)


# In[19]:


plt.scatter(y_test, result)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")


# This time , the predicted vs actual values is giving a leaner graph, which is better. Lets look further into R2 score and MSE.

# In[20]:


print('R2 score : ',metrics.r2_score(y_test, result))
print('Variance: ',metrics.explained_variance_score(y_test,result))
print('MSE ', metrics.mean_squared_error(y_test,result))


# Addition of the column 'Avg. Session Length' has greatly improved the model for us with increased R2 score of 0.981 and reduced MSE of 118.68
