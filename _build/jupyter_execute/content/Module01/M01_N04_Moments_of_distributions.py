#!/usr/bin/env python
# coding: utf-8

# # Moments of distributions
# ***Reading: Emile-Geay: Chapter 3***
# 
# "Climate is what you expect. Weather is what you get"
# 
# "Expectaion is what you expect. The random variable is what you get" 

# In[ ]:


get_ipython().run_line_magic('reset', '')
import numpy as np
import matplotlib.pyplot as plt


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


# # 1. Moments of distributions
# ## 1.1 Expected value/mean
# 
# The expected value of a random variable is the average value we would expect to get if we could sample a random variable an infinite number of times. It represents a average of all the possible outcomes, weighted by how probably they are. 
# 
# The *expected value* of a random variable is also called its *first order moment*, *mean*, or *average*. It is computed using the *expectation operator*:
# $$E(X)=\mu=\begin{cases}
# \sum_{i=1}^{N}x_{i}P(X=x_{i}) & \text{if }X\text{ is discrete}\\
# \int_{\mathbb{R}}xf(x)dx & \text{if }X\text{ is continuous}
# \end{cases}$$
# 
# **Key property: linearity**
# 
# $$E(aX+bY)=aE(x)+bE(y)$$
# 
# We can also define the expected value, or mean, of any function of a random variable:
# $$E(g(X))=\begin{cases}
# \sum_{i=1}^{N}g(x_{i})P(X=x_{i}) & \text{if }X\text{ is discrete}\\
# \int_{\mathbb{R}}g(x)f(x)dx & \text{if }X\text{ is continuous}
# \end{cases}$$
# 
# 

# ##  1.2 Higher Order Moments
# We can define higher order moments of a distribution as
# $$ m(X,n)=E(X^n)=\sum_{i=1}^N x_i^nP(X=x_i)$$
# $$ m(X,n)=E(X^n)=\int_\mathbb{R}xf(x)dx$$
# for, respectively, discrete and continuous r.v.s
# 
# ##  1.3 Variance
# A closely related notion to the second order moment is the **variance** or centered second moment, defined as:
# $$V(X)=E([X-E(x)]^2)=E([X-\mu]^2)=\int_\mathbb{R}(x-\mu)^2f(x)dx$$
# 
# Expanding the square and using the linearity of the expectation operator, we can show that the variane can also be written as:
# $$V(X)=E(X^2)-(E(X))^2=E(X^2)-\mu^2$$
# 
# Variance is a measure of the spread of a distribution. 
# 
# ## 1.4 Standard deviation 
# Another closely related measure is standard deviation, devined simply as the square root of the variance
# $$\text{std}=\sqrt{V(X)}=\sqrt{E([X-\mu]^2)}$$
# 
# 
# ### Important Properties:
# $$ V(X+b)=V(x)$$
# $$ V(aX)=a^2V(x)$$
# $$ \text{std}(aX)=a \cdot\text{std}(X)$$
# 

# ## 1.4 Examples
# ### Uniform distributions
# The pdf of a r.v. uniformly distributed over the interval $[a,b]$ is
# $$f(x)=\frac{1}{b-a}$$
# You can check yourselves that 
# $$ E(X)=\frac{1}{2}(a+b)$$
# $$ V(X)=\frac{1}{12}(b-a)^2$$
# $$\text{std}=\frac{1}{\sqrt{12}}(b-a)$$
# 
# ### Normal distribution
# The pdf of a normally distributed r.v. with location parameter $\mu$ and scale parameter $\sigma$ is
# $$f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]$$
# You can check yourselves that 
# $$ E(X)=\mu$$
# $$ V(X)=\sigma^2$$
# $$\text{std}=\sigma$$
# 
# ![image.png](attachment:image.png)

# 
# # 2. Law of large numbers
# 

# In[1]:


from scipy import stats
import numpy as np


mu=2;
sigma=5;

# you should also play arond with the number of draws and bins of the histogram.
# there are some guidelines for choosing the number of bins (Emile-Geay's book talks a bit about them)

Ndraws=100000000;


# generate random variables and define edges (note we want the integers to be in the bins, not at the edges)
X_norm=stats.norm.rvs(loc=mu,scale=sigma, size=Ndraws)

print(np.mean(X_norm))
print(np.abs(np.mean(X_norm)-mu))


# In[ ]:




