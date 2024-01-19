#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


# In[43]:


def norm(x):
    return (x-np.mean(x))/np.std(x)

n_samples=5000

t=np.arange(n_samples)
x1=stats.norm.rvs(loc=0,scale=1,size=n_samples)
x2=stats.norm.rvs(loc=0,scale=1,size=n_samples)
x3=stats.norm.rvs(loc=0,scale=1,size=n_samples)
y=x1*x2*x3

X=np.stack([norm(x1),norm(x2),norm(x3)],axis=1);


# In[44]:


fig,ax=plt.subplots(2,2,figsize=[20,12])
ax[0,0].plot(t,norm(x1)-5,label='$x_1$')
ax[0,0].plot(t,norm(x2)-10,label='$x_2$')
ax[0,0].plot(t,norm(x3)-15,label='$x_3$')
ax[0,0].plot(t,norm(y),linewidth=2,label='$y$')
ax[0,0].legend(ncol=4)
ax[0,0].set_xlim(0,200)


ax[0,1].plot(x1,norm(y),'o')

ax[1,0].plot(x2,norm(y),'o')
ax[1,1].plot(x3,norm(y),'o')


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)


# In[46]:





# In[53]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression().fit(Xn,yn)

yhat_train_0=lin_reg.predict(X_train);
yhat_test_0=lin_reg.predict(X_test);


from sklearn.neural_network import MLPRegressor

model_1 = MLPRegressor(hidden_layer_sizes=5,max_iter=5000)
model_2 = MLPRegressor(hidden_layer_sizes=16,max_iter=5000)
model_3 = MLPRegressor(hidden_layer_sizes=[64,64],max_iter=500)


mlp_1=model_1.fit(X_train, y_train)
mlp_2=model_2.fit(X_train, y_train)
mlp_3=model_3.fit(X_train, y_train)

yhat_train_1=mlp_1.predict(X_train);
yhat_train_2=mlp_2.predict(X_train);
yhat_train_3=mlp_3.predict(X_train);


yhat_test_1=mlp_1.predict(X_test);
yhat_test_2=mlp_2.predict(X_test);
yhat_test_3=mlp_3.predict(X_test);


# In[52]:


fig,ax=plt.subplots(2,4,figsize=[20,12])

for i in [0,1]:
    for j in [0,1,2,3]:
        ax[i,j].plot([np.min(yn),np.max(yn)],[np.min(yn),np.max(yn)],'k-')
        ax[i,j].set_xlabel('$y$')

ax[0,0].set_ylabel('$\hat y$')        

ax[0,0].plot(y_train,yhat_train_0,'r.')
ax[0,1].plot(y_train,yhat_train_1,'r.')
ax[0,2].plot(y_train,yhat_train_2,'r.')
ax[0,3].plot(y_train,yhat_train_3,'r.')

ax[1,0].plot(y_test,yhat_test_0,'r.')
ax[1,1].plot(y_test,yhat_test_1,'r.')
ax[1,2].plot(y_test,yhat_test_2,'r.')
ax[1,3].plot(y_test,yhat_test_3,'r.')



# In[9]:


mlp_3.get_params()


# In[ ]:


mlp_1.intercepts_


# In[491]:


mlp_1.coefs_


# In[434]:


mlp_1.n_features_in_


# In[ ]:





# In[110]:


import numpy as np
x=np.linspace(-20,20,100)
y=np.sinc(x)
plt.plot(x,y)


# In[143]:


def norm(x):
    return (x-np.mean(x))/np.std(x)


n_samples=100000
x=stats.uniform.rvs(loc=-5,scale=10,size=n_samples)
y=np.sin(x*2)
X=x.reshape(-1,1)
plt.plot(X,y,'.')


# In[144]:


from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=[64,64,64],max_iter=5000)
mlp   =model.fit(X,y)
yhat  =model.predict(X)


# In[145]:


x1=np.linspace(-5,5,200).reshape(-1,1)
x2=np.linspace(-7,7,1000).reshape(-1,1)
y1=model.predict(x1)
y2=model.predict(x2)


# In[146]:


fig,ax=plt.subplots(1,figsize=[12,6])
plt.plot(x,y,'ok',label='training data')
plt.plot(x2,y2,'r-',linewidth=3,label='Neural Net')    
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


# In[ ]:





# In[ ]:




