#!/usr/bin/env python
# coding: utf-8

# # Regression

# In[1]:


get_ipython().run_line_magic('reset', '')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

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


# In[2]:


# Define the true process
nsample = 100
beta_true = np.array([10, 1])
var_eps=3

# generate data
x = np.linspace(0, 10, nsample)
e = stats.norm(loc=0,scale=var_eps).rvs(size=nsample)
#e = stats.cauchy(loc=0,scale=1).rvs(size=nsample)
y_true=np.dot(X, beta_true)
y = y_true + e


print('R2_true=',np.var(y_true)/(np.var(y_true)+var_eps))
print('beta_0=',beta_true[0],',  beta_1=',beta_true[1])
print(res.summary())


fig,ax=plt.subplots(1,1,figsize=[12,12])

ax.plot(x,y,'o')
ax.set_xlabel('x')
ax.set_ylabel('y')
# Define the true process
nsample = 100
beta_true = np.array([10, 1])
var_eps=3


x = np.linspace(0, 10, nsample)
e = stats.norm(loc=0,scale=var_eps).rvs(size=nsample)
X = sm.add_constant(x)
y_true=np.dot(X, beta_true)
y = y_true + e


# In[49]:


# Define the true process
nsample = 100
beta_true = np.array([10, 1.2])
var_eps=3

# generate data
x = np.linspace(0, 10, nsample)
#e = stats.norm(loc=0,scale=var_eps).rvs(size=nsample)
e = stats.cauchy(loc=0,scale=0.5).rvs(size=nsample)

X = sm.add_constant(x)
y_true=np.dot(X, beta_true)
y = y_true + e


print('R2_true=',np.var(y_true)/(np.var(y_true)+var_eps))
print('beta_0=',beta_true[0],',  beta_1=',beta_true[1])

model = sm.OLS(y, X)
res = model.fit()
print(res.summary())


st, data, ss2 = summary_table(res, alpha=0.05)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T


fig,ax=plt.subplots(1,1,figsize=[12,12])
pred_ols = res.get_prediction()
#iv_l = pred_ols.summary_frame()["obs_ci_lower"]
#iv_u = pred_ols.summary_frame()["obs_ci_upper"]

ax.plot(x, y,'o', label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, fittedvalues, 'r-', label='OLS')
ax.plot(x, predict_ci_low, 'r--')
ax.plot(x, predict_ci_upp, 'r--',label='5-95% ci on values')
ax.plot(x, predict_ci_upp, 'r--')
ax.plot(x, predict_mean_ci_low, 'r:',label='5-95% ci on mean')
ax.plot(x, predict_mean_ci_upp, 'r:')
ax.legend(loc="best")


# In[48]:


fig,ax=plt.subplots(1,1,figsize=[8,8])

x=np.linspace(0.2,1,200)
beta_hat=res.params[1]

pdf=stats.norm.pdf(x,loc=beta_hat,scale=res.bse[1])
ax.plot(x,pdf,label='uncertainty range')
ax.vlines(beta_hat,0,np.max(pdf),label='best estimate')
ax.vlines(beta_true[1],0,np.max(pdf),'r',label='true beta')
ax.legend()
ax.grid()


# In[39]:


Ndraws=100000
q=np.zeros(Ndraws)

for n in range(Ndraws):

    # generate data
    x = np.linspace(0, 10, nsample)
    #e = stats.norm(loc=0,scale=var_eps).rvs(size=nsample)
    e = stats.cauchy(loc=0,scale=1).rvs(size=nsample)
    X = sm.add_constant(x)
    y_true=np.dot(X, beta_true)
    y = y_true + e

    model = sm.OLS(y, X)
    res = model.fit()

    beta_hat=res.params[1]
    stderr=res.bse[1]
    q[n]=stats.norm.cdf(beta_true[1], loc=beta_hat, scale=stderr)


# In[42]:


np.shape(q[q<0.05])[0]/Ndraws*100


# In[50]:


Ndraws=1000
q=np.zeros(Ndraws)
beta_true[1]=0
for n in range(Ndraws):
    # generate data

    x = np.linspace(0, 10, nsample)
    e = stats.norm(loc=0,scale=20).rvs(size=nsample)
    #e = stats.cauchy(loc=0,scale=1).rvs(size=nsample)
    X = sm.add_constant(x)
    y_true=np.dot(X, beta_true)
    y = y_true + e

    model = sm.OLS(y, X)
    res = model.fit()

    beta_hat=res.params[1]
    stderr=res.bse[1]
    q[n]=beta_hat


# In[ ]:


q


# In[56]:


# Define the true process
nsample = 30
beta_true = np.array([10, 0])
var_eps=3

# generate data
x = np.linspace(0, 10, nsample)
e = stats.norm(loc=0,scale=var_eps).rvs(size=nsample)
#e = stats.cauchy(loc=0,scale=1).rvs(size=nsample)

X = sm.add_constant(x)
y_true=np.dot(X, beta_true)
y = y_true + e


print('R2_true=',np.var(y_true)/(np.var(y_true)+var_eps))
print('beta_0=',beta_true[0],',  beta_1=',beta_true[1])

model = sm.OLS(y, X)
res = model.fit()
#print(res.summary())


st, data, ss2 = summary_table(res, alpha=0.05)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T


fig,ax=plt.subplots(1,1,figsize=[12,12])
pred_ols = res.get_prediction()
#iv_l = pred_ols.summary_frame()["obs_ci_lower"]
#iv_u = pred_ols.summary_frame()["obs_ci_upper"]

ax.plot(x, y,'o', label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, fittedvalues, 'r-', label='OLS')
ax.plot(x, predict_ci_low, 'r--')
ax.plot(x, predict_ci_upp, 'r--',label='5-95% ci on values')
ax.plot(x, predict_ci_upp, 'r--')
ax.plot(x, predict_mean_ci_low, 'r:',label='5-95% ci on mean')
ax.plot(x, predict_mean_ci_upp, 'r:')
ax.legend(loc="best")


# In[57]:


Ndraws=10000
q=np.zeros(Ndraws)
beta_true[1]=0
for n in range(Ndraws):
    # generate data

    x = np.linspace(0, 10, nsample)
    e = stats.norm(loc=0,scale=20).rvs(size=nsample)
    #e = stats.cauchy(loc=0,scale=1).rvs(size=nsample)
    X = sm.add_constant(x)
    y_true=np.dot(X, beta_true)
    y = y_true + e

    model = sm.OLS(y, X)
    res = model.fit()

    beta_hat=res.params[1]
    stderr=res.bse[1]
    q[n]=beta_hat


# In[58]:


plt.hist(q,50)


# In[ ]:


nsample=10

Ndraws=10000
q=np.zeros(Ndraws)
for n in range(Ndraws):
    e1 = stats.norm(loc=0,scale=10).rvs(size=nsample)
    e2 = stats.norm(loc=0,scale=10).rvs(size=nsample)
    q[n]=np.mean(e1)-np.mean(e2)


# In[ ]:


plt.hist(q,50)


# In[ ]:


q


# In[ ]:




