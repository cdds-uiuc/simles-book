#!/usr/bin/env python
# coding: utf-8

# # Regression

# In[54]:


get_ipython().run_line_magic('reset', '')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression

# These are some parameters to make figures nice (and big)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['figure.figsize'] = 16,8
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


# In[138]:


#Generate some data:
Npoints=100;
beta=3
sigma_e=10;
x=np.linspace(0,10,Npoints).reshape(-1,1)


Nrealizations=1000


eps=stats.norm.rvs(loc=0,scale=sigma_e,size=Npoints).reshape(-1,1)
y_tilde=beta*x+eps


# fit the data and obtain a prediction yhat
reg=LinearRegression().fit(x.reshape(-1,1),y_tilde)
yhat=reg.predict(x);
r2=reg.score(x,y_tilde)

fig, ax = plt.subplots(1, 1,figsize=[8,8])
ax.plot(x,y_tilde)
ax.plot(x,yhat,'r')

print(np.var(beta*x))
print(r2)


# In[93]:


np.shape(y_tilde)


# In[89]:


x=np.linspace(0,10,Npoints,axis=-1).reshape(-1,1)
np.shape(x)
plot(x,y)


# In[75]:





# In[ ]:




