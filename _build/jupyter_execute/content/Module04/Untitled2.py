#!/usr/bin/env python
# coding: utf-8

# K-means clustering

# In[66]:


import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
mux_1=1
mux_2=4
mux_3=6

muy_1=3
muy_2=2
muy_3=1

muz_1=3
muz_2=1
muz_3=5

sigx_1=0.25
sigx_2=0.25
sigx_3=0.25


sigy_1=0.25
sigy_2=0.25
sigy_3=0.25

sigz_1=0.25
sigz_2=0.25
sigz_3=0.25


size=30

x1=stats.norm.rvs(loc=mux_1,scale=sigx_1,size=size)
x2=stats.norm.rvs(loc=mux_2,scale=sigx_2,size=size)
x3=stats.norm.rvs(loc=mux_3,scale=sigx_3,size=size)

y1=stats.norm.rvs(loc=muy_1,scale=sigy_1,size=size)
y2=stats.norm.rvs(loc=muy_2,scale=sigy_2,size=size)
y3=stats.norm.rvs(loc=muy_3,scale=sigy_3,size=size)

z1=stats.norm.rvs(loc=muz_1,scale=sigz_1,size=size)
z2=stats.norm.rvs(loc=muz_2,scale=sigz_2,size=size)
z3=stats.norm.rvs(loc=muz_3,scale=sigz_3,size=size)


x=np.concatenate([x1,x2,x3])
y=np.concatenate([y1,y2,y3])
z=np.concatenate([z1,z2,z3])


# In[67]:


fig,ax = plt.subplots(1,2,figsize=[20,10])
ax[0].plot(x,np.ones(np.shape(x)),'o')
ax[1].plot(x1,np.ones(np.shape(x1)),'o')
ax[1].plot(x2,np.ones(np.shape(y1)),'o')
ax[1].plot(x3,np.ones(np.shape(z1)),'o')


# In[68]:


fig,ax = plt.subplots(1,2,figsize=[20,10])
ax[0].plot(x,y,'o')
ax[1].plot(x1,y1,'o')
ax[1].plot(x2,y2,'o')
ax[1].plot(x3,y3,'o')


# In[69]:


fig = plt.figure(figsize=[9,9])
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z)


# In[65]:


fig,ax = plt.subplots(ncols=2, subplot_kw={'projection': '3d'},figsize=[20,10])
ax[0].plot(x,y,z,'o')
ax[1].plot(x1,y1,z1,'o')
ax[1].plot(x2,y2,z2,'o')
ax[1].plot(x3,y3,z3,'o')


# In[ ]:




