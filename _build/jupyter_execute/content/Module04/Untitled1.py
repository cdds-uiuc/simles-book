#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
from matplotlib import pyplot as plt
N=1000;
x=np.arange(0,N)
y=x+np.random.normal(size=N)

x=np.random.normal(size=N)
y=np.random.normal(size=N)


# In[42]:


plt.subplots(1,figsize=[20,20])
#plt.plot(x,y,'o',color='#8F2727',markersize=20)
plt.plot(x,y,'o',color='#8F2727F2',markersize=40)


# In[ ]:




