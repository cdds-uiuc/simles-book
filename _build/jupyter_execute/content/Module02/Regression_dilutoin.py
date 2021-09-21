#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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


# **Truth**:<br>
# $ y_0=\beta x_0$<br>
# 
# **Observe**:<br>
# $y=y_0 +\epsilon_y$<br>
# $x=x_0 +\epsilon_x$<br>
# 
# **Naive** <br>
# $\hat \beta=\frac{\overline {xy}}{\overline{xx}}$

# In[17]:


# Define the true process
nsample = 100
beta   = 3
var_y=3
var_x=5;

e_x=stats.norm(loc=0,scale=var_x).rvs(size=nsample)
e_y=stats.norm(loc=0,scale=var_y).rvs(size=nsample)

# generate data
x0 = np.linspace(0, 10, nsample)
y0 = beta*x0
y  = y0+e_y
x  = x0+e_x

#fit 
beta_hat=np.sum(x*y)/np.sum(x*x)
y_hat=beta_hat*x

fig,ax=plt.subplots(1,1,figsize=[10,10])
ax.plot(x0,y0,'k',label='true')
ax.plot(x,y_hat,'r',label='estimate')
ax.plot(x0,y,'bo')
ax.plot(x,y,'ro')
ax.legend()
#ax.set_xlim(-5,15)
#ax.set_ylim(-30,70)


# In[ ]:




