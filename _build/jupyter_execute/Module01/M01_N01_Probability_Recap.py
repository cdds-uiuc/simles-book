#!/usr/bin/env python
# coding: utf-8

# # Basics of probability
# 

# ## Fair Dice
# 
# We'll start with a quick recap of notions of probability, using a pair of fair dice as an example. We'll define some notions of probability and discuss them in the context of a pair of dice. Let's call the first dice *X* and the second dice *Y*. The total roll of the double dice is *D*. The values of *D* are in the table below!
# ![Screen%20Shot%202021-08-23%20at%204.05.39%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%204.05.39%20PM.png)
# 
# 
# <html><span>&#9856</span></html>
# 
# ---

# 

# 
# ### **Sample Space**: *S*
#      
#      The set *S* of all possible outcomes of a trial or experiment. It can be discrete or continuous. 
#      
#      For a single dice (let's say dice x), the sample space is the set $$S_X=\left\{1,2,3,4,5,6\right\}$$ 
#      
#      For rolling the two dice $X$, and $Y$ the sample space the full set of possible combinations
# 
# $$ S_{X,Y}=\left\{\begin{split} & (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),\\ 
# & (2,1),(2,2),(2,3),(2,4),(2,5),(2,6)\\
# & (3,1),(3,2),(3,3),(3,4),(3,5),(3,6)\\
# & (4,1),(4,2),(4,3),(4,4),(4,5),(4,6)\\
# & (5,1),(5,2),(5,3),(5,4),(5,5),(5,6)\\
# & (6,1),(6,2),(6,3),(6,4),(6,5),(6,6)\\
# \end{split}\right\} $$
# 
# 
# ---

# 

# ### **Event**: *E* 
# 
# Some subset *E* of the sample space. For example, the event "dice rolled a 1" would be the subset E*={1} of the sample space. The event "the dice rolled an odd number" would be the subset *E*={1,3,5}.  
# The event "the first dice rolled a 3, and the second dice rolled a 4" would be $E=\left\{(3,4)\right\}$.
#     
# ![Screen%20Shot%202021-08-23%20at%204.14.27%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%204.14.27%20PM.png)

#   Be careful here. Contrast this with the event "both a 3 and a 4 were rolled", which doesn't specify the order. For this event, the subset is $$E=\left\{(3,4),(4,3)\right\}$$
#   
# ![Screen%20Shot%202021-08-23%20at%204.14.31%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%204.14.31%20PM.png)
# 
# 

# The event "a total of seven was rolled" is $$E=\left\{ (X,Y):X+Y=7\right\}==\left\{(1,6),(6,1),(2,5),(5,2),(3,4),(4,3)\right\}$$
# ![Screen%20Shot%202021-08-23%20at%204.14.36%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%204.14.36%20PM.png)
# ---

# ### **Probability of event $E$** : $P(E)$
#  
#  $$P(E)=\frac{\text{Number of ways E can occur }}{\text{ Total number of possible outcomes }}$$
# 
# This is the same as the size (or measure) $|\cdot |$of the subset $E$ representing that event relative to the size of the sample set $S$. 
#     
# $$P(E) =\frac{|E|}{|S|}$$
# For a dice the probability of rolling 1 is equal to
#     
# $$P(E)=\frac{\text{Number of ways "rolling 1" can occur }}{\text{ Total number of possible outcomes }}=\frac{|\left\{1\right\}|}{|S_X|}=1/6$$
#     
# The probability of rolling "rolling an odd number" is equal to
#     
# $$P(E)=\frac{\text{Number of ways "rolling an odd number" can occur }}{\text{ Total number of possible outcomes }}=\frac{|\left\{1,3,5\right\}|}{|S_X|}=3/6=1/2$$
# 
# Finally, the probability of rolling a combined 7 is 
# $$P(E)=\frac{\text{Number of ways "rolling a sece" can occur }}{\text{ Total number of possible outcomes }}=\frac{|\left\{(1,6),(6,1),(2,5),(5,2),(3,4),(4,3)\right\}|}{|S_{X,Y}|}=\frac{6}{36}=1/6$$
# ---

# ### Joint Probability
# A joint probability is the probability that two events $E_1,E_2$ both happen.We would denote this $P(E_1,E_2)$, or $P(E_1\land E_2)$, where $\land$ represents the boolean operation `AND`.  
# 
# A simple example is the probability that $E_1$: Dice $X$ rolls a 1 and event $E_2$,dice $Y$ rolls a 3. In this case, 
# ![Screen%20Shot%202021-08-23%20at%208.21.58%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%208.21.58%20PM.png)
# 
# 

# You can see that the probability of both events happening is the same size as the intersection of the sets 
# corresponding to each event. $$P(E_1,E_2)=P(E_1\cap E_2)$$
# ![Screen%20Shot%202021-08-23%20at%208.22.21%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%208.22.21%20PM.png)
# $$P(E_1,E_2)=P(E_1\cap E_2)=\frac{|\{(1,3)\}|}{|S_{X,Y}|}=\frac{1}{36}$$  
# 
# ---

# 
# Here is another example: $E_1$: 'the total roll is $D=6$ and $E_2$: 'dice $X$ rolled an odd roll'.
# The probability of this joint event is the size of the subset that satisifies both events, i.e. the intersection of the two sets $E_1$ and $E_2$
# ![Screen%20Shot%202021-08-23%20at%208.16.05%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%208.16.05%20PM.png)
# 
# 

# The intersection is depicted below:
#  
# ![Screen%20Shot%202021-08-23%20at%208.18.22%20PM.png](attachment:Screen%20Shot%202021-08-23%20at%208.18.22%20PM.png)
# 
# 
# $$P(E_1,E_2)=P(E_1\cap E_2)=\frac{|\{(1,5),(5,1),(3,3)\}|}{|S_{X,Y}|}=\frac{3}{36}=\frac{1}{12}$$

# ### Independent Events 
# Events $E_1$ and $E_2$ are independt if $P(E_1,E_2)=P(E_1)P(E_2)$, that is , if the probability of Event $E_1$ happenign is not affected by whether Event $E_2$ happens. Going back to thinking of sets as events, this happens if $$P(E_1,E_2)=P(E_1)\cdot P(E_2)$$
# 
# For example, the events $E1$ : Dice $X$ rolls a 1 and event $E_2$,dice $Y$ rolls a 3 are clearly independent. Whether $Y$ has rolled a 3 or not has no bearing on what $X$ rolls. Consequently, 
# $$P(E_1,E_2)=P(E_1\cap E_2)=\frac{1}{36}=\frac{1}{6}\frac{1}{6}=\frac{|\{1\}|}{|S_X|}\frac{|\{3\}|}{|S_Y|}=P(E_1)P(E_2)$$
# 
# On the other hand, the two events $E_1$ 'the total roll is $D$=6 and $E_2$: 'dice $X$ rolled an odd roll' are not independent. If $X$ rolled an odd roll, there are three combinations that will yield 6. If, on the other hand $X$ has rolled an even roll (i.e. either 2,4, or 6), there are only two combinations that will yield 6.
# 
# In this case, $P(E_1, E_2)=\frac{3}{36}\neq P(E_1)P(E_2)=\frac{5}{36}\frac{18}{36}=\frac{2.5}{36}$. 
# 
# ---

# ### Conditional Probability
# 
# Conditional probability is the probabiltiy that event $E_1$  occurs ***given*** that event $E_2$ has occured. For example, the probability of $E_1$ 'the total roll is $D=6$ given that $E_2$: '$X$ rolled an odd roll' has occured.
# 
# Given that $E_2$ has occured the sample space is now only the subset where X is odd. This subspace has 18 elements. Out of these, $E_1$ corresponds to the subset $\{ (1,5),(3,3),(5,1)\}$. So the probability of $D=6$ ***given that *** X is odd is denoted as: $P(E_1|E_2)=3/18=1/6$. 
# 
# Conditional probabilities are formally defined as:
# $$P(E_1|E_2)=\frac{P(E_1, E_2)}{P(E_2)}$$.
# 
# For our eample we can verif this. We've calculated the joint probability as 3/36, and the probability of '$X$ rolls odd is 1/2
# 
# $$P(E_1|E_2)=\frac{P(E_1,E_2)}{P(E_2)}=\frac{3/36}{1/2}=1/6$$.
# 
# ---

# ### Bayes Rule
# If we rewroite the definition of conditional probability as $P(E_1,E_2)=P(E_1|E_2)P(E_2)$, and take advantage of the commutativity of the intersection operator, i.e. that $P(E_1,E_2)=P(E_1\cap E_2)=P(E_2 \cap E_1)$, we can write
# 
# $$P(E_1,E_2)=P(E_1|E_2)P(E_2) = P(E_2|E_1)P(E_1)$$
# 
# and immediately derive Bayes rule (Which will come in very handy later in the course)
# 
# $$ P(E_1|E_2)=\frac{P(E_2|E_1)P(E_1)}{P(E_2)}$$
# 
# 
# ---

# ### Law of Total probability
# 
# The law of total probability says that if we have a partition of the sample space, $B_n$ such that $B_i\cap B_j=\phi$ if $i\neq j$.  and $$\cap_{n} B_n = S$$, then
# 
# $$P(E)=\sum_n P(E|B_n)P(B_n)$$
# 
# This should be intuitive with the fair dice example. For example, let $E$ be the event 'A total roll $D=6$ was rolled'. A partition $B_n$ could be 'the dice $X$ rolled n' for $n$ between 1 and 6. Thus, the total probability of $D=6$ is the sum of the probability of rolling a seven given that $X$ rolled 1, plus the probability of rolling a seven given that $X$ rolled a 2, and so on....
# 
# ---

# <hr style="border:2px solid black
#            "> </hr>
# 

# # 1.2 Random variables
# 
# **A random variable is a Real-valued variable whose whose values depend on outcomes of a random phenomenon.**
# 
# Consider the case of a single fair dice, with possible values, i.e. sample space: $$S=\{1,2,3,4,5,6\}$$ 
# We can define a random variable $X$ whose value is equal to the dice roll. This random variable could take ***discrete*** values between 1 and 6.
# 
# 
# 
# ## Probability mass function
# 
# The **probability mass function** , $p$ of a random variable , $X$ is the probability that the variable takes on a given value, and is often denoted by a small cap $p$. For example, the probability that $X$ takes the value 5, could be denoted as $p(X=5)$ or $p_X(5)$ or, simply $p(5)$. 
# 
# Our $X$ takes discrete values between 1 and 6, and the probabilit is the same for each value. We would call this a uniformly distributed discrete random variable, or a random variable with a discrete uniform distribution between 1 and 6.
# 
# Let's generate such a variable in python:

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# the stats class has a method to generate randint objects that deal with uniform discrete (integer-valued) random variables. 
# the randint object itself has method rvs to generate such a random variable:


#rerun this cell several times
X=stats.randint.rvs(low=1, high=7, size=1)
print(X)


# ## Probability vs event
# 
# At this point it may seem like I'm inventing new terminology. For example, why do we need to call $D$ a random variable, and talk about the possibility that it takes on different values? It seems like the probability of X taking on each value is just the probability of each event in $S$. 
# 
# Here is another example of a random variable on the same sample space: $Z$ is a random variable which takes the value $Z=0$ if the dice roll is odd and $Z=1$ if the dice roll is even. Thus, even though $X$ and $Z$ are associated with the same sample space and events, they take on different values. 
# 
# In this case, since $Z$ takes on only two values, 0 and 1 $Z$ would be called a Bernoulli random variable.
# 
# 

# In[53]:


#rerun this cell several times
Z=stats.bernoulli.rvs(0.5, size=1)
print(Z)


# # Why are we doing this? 
# 
# While the dice example here seems so simple as to be useless, it illustrates several properties of probabilities that are going to be useful later on when we're talking about random variables:
# 
# * ** Conditional Probability** The probability of random variable $X$ taking the value $a$, conditional on random variable $Y$ taking value $b$ is:
# $$p(X=a|Y=b)=\frac{p(X=a,Y=b)}{p(Y=b)}$$
# 
# 
# * **Independence**: if random variables are independent, their joint is the product of their individual probability  $$p(X=a,Y=b)=p_{X,Y}(a,b)=p_X(a)p_Y(b)$$
# 
# and so on for Bayes' rule and the law of total probability 
