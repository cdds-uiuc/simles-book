#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
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
plt.rcParams.update({'font.size': 16})


# In[2]:


def logistic(x,beta):
    return np.exp(beta[0]+x*beta[1])/(1+np.exp(beta[0]+x*beta[1]))


# # Generate Data

# In[28]:


ndata=500
mu=[0]
scale=[1000]
beta=[-1000/50,1/50]

x=stats.expon.rvs(loc=0,scale=scale[0],size=ndata)
y=stats.bernoulli.rvs(p=logistic(x,beta))


# In[29]:


plt.plot(x,logistic(x,beta),'.')


# In[30]:


plt.hist(x)


# In[31]:


fig,ax=plt.subplots(figsize=[10,5])

# generate some data
#for j in range(ndata)
ind=np.arange(ndata)
plt.plot(ind[y==0],x[y==0],'o',markersize=10)
plt.plot(ind[y==1],x[y==1],'o',markersize=10,label='stormy')
plt.xlabel('day');
plt.ylabel('cape');
plt.legend()
plt.ylim(0,5000)


# In[32]:


fig,ax=plt.subplots(figsize=[20,10])
plt.plot(x,y,'o')
plt.xlabel('cape')
plt.ylabel('Storm (1=yes)')


# In[33]:


z=stats.binned_statistic(x, y, statistic='mean', bins=100)


# In[34]:


t=np.arange(0,7000)
fig,ax=plt.subplots(figsize=[10,5])
plt.plot(x,y,'o')
plt.vlines(z.bin_edges,0,1,linestyle='--',color=[0.8,0.8,0.8])
bins=0.5*(z.bin_edges[1:]+z.bin_edges[:-1])
plt.plot(bins,z.statistic,'o',label='bin-average freq of storms')
plt.plot(t,logistic(t,beta),'-',label='logistic function')
plt.legend()
plt.xlabel('cape')
plt.ylabel('Storm (1=yes)')
plt.xlim(0,3000)


# In[36]:


z=stats.binned_statistic(x, y, statistic='mean', bins=50)
K=150
t=np.arange(0,7000)
fig,ax=plt.subplots(figsize=[10,5])
plt.plot(x/K,y,'o',label='Label')
plt.vlines(z.bin_edges/K,0,1,linestyle='--',color=[0.8,0.8,0.8])
bins=0.5*(z.bin_edges[1:]+z.bin_edges[:-1])
plt.plot(bins/K,z.statistic,'o',label='bin-average frequency')
plt.plot(t/K,logistic(t,beta),'-',label='logistic function')
plt.legend()
plt.xlabel('cape')
plt.ylabel('Storm (1=yes)')
plt.xlim(0,40)


# In[125]:


bins=0.5*(z.bin_edges[1:]+z.bin_edges[:-1])
plt.plot(bins,z.statistic,'o')


# In[85]:


#plt.plot(bins,z.statistic,'o-',label='freq of storms')
fig,ax=plt.subplots(figsize=[20,10])
plt.plot(z.statistic,logistic(bins,beta),'o',label='logistic function',markersize=20)
plt.xlabel('predicted probability')
plt.ylabel('observed frequency')
plt.plot([0,1],[0,1],'k--')


# In[64]:


`


# In[ ]:




