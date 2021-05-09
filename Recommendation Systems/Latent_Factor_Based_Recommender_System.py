#!/usr/bin/env python
# coding: utf-8

# # Week 2: Latent Factor-Based Recommender Systems
# 
# Last week we went over some basics of Recommender Systems for similarity based recommendations. In this notebook we will learn the basics of Latent Factor Models, as well as how to implement them.

# ## Part 1: Setting up the Data
# 
# This week we will be using another amazon review dataset, this time the dataset is about Watches. 
# https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Watches_v1_00.tsv.gz

# In[1]:


#TODO Set up our dataset how we have in the past, after importing our typical imports, get a header, create a dataset and
#  fill the dataset, appropriately int casting our rating/vote values


#Start with our typical imports
import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import numpy
import random


# In[2]:


#We will need these dictionaries down below, Lets create them now
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)


# ## Part 2: Simple Latent Factor-Based Recomender
# 
# Here we'll use gradient descent to implement a machine-learning-based recommender (a latent-factor model).
# 
# This is a fairly difficult exercise, but brings together many of the ideas we've seen previously in this class, especially regarding gradient descent. This will be a relatively light notebook given this case, but you will need to know how to do this __on your own__ for your capstone project!
# 
# First, we build some utility data structures to store the variables of our model (alpha, userBiases, and itemBiases)

# In[3]:


#Getting the respective lengths of our dataset and dictionaries
N = len(dataset)
nUsers = len(reviewsPerUser)
nItems = len(reviewsPerItem)

#Getting a list of keys
users = list(reviewsPerUser.keys())
items = list(reviewsPerItem.keys())

#This is equivalent to our Rating Mean from week 1
alpha = sum([d['star_rating'] for d in dataset]) / len(dataset)

#Create another two defaultdict's, this time being float types because they are prediction based
userBiases = defaultdict(float)
itemBiases = defaultdict(float)

#Can't forget our MSE function
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# The actual prediction function of our model is simple: Just predict using a global offset (alpha), a user offset (beta_u in the slides), and an item offset (beta_i)

# In[4]:


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]


# We'll use another library in this example to perform gradient descent. This library requires that we pass it a "flat" parameter vector (theta) containing all of our parameters. This utility function just converts between a flat feature vector, and our model parameters, i.e., it "unpacks" theta into our offset and bias parameters.

# In[5]:


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))


# The "cost" function is the function we are trying to optimize. Again this is a requirement of the gradient descent library we'll use. In this case, we're just computing the (regularized) MSE of a particular solution (theta), and returning the cost.

# In[6]:


def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['customer_id'], d['product_id']) for d in dataset]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost


# The derivative function is the most difficult to implement, but follows the definitions of the derivatives for this model as given in the lectures. This step could be avoided if using a gradient descent implementation based on (e.g.) Tensorflow.

# In[7]:


def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(dataset)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for d in dataset:
        u,i = d['customer_id'], d['product_id']
        pred = prediction(u, i)
        diff = pred - d['star_rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return numpy.array(dtheta)


# Compute the MSE of a trivial baseline (always predicting the mean) for comparison:

# In[8]:


alwaysPredictMean = [alpha for d in dataset]
labels = [d['star_rating'] for d in dataset]

MSE(alwaysPredictMean, labels) #Should be 1.6725...


# Finally, we can run gradient descent. This particular gradient descent library takes as inputs (1) Our cost function (implemented above); (2) Initial parameter values; (3) Our derivative function; and (4) Any additional arguments to be passed to the cost function (in this case the labels and the regularization strength).

# In[9]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, 0.001))


# ## Part 3: Complete Latent Factor Model
# 
# 

# For each user and item we now have a low dimensional descriptor (which represents a user's preferences), of dimension K.

# In[10]:


userBiases = defaultdict(float)
itemBiases = defaultdict(float)
userGamma = {}
itemGamma = {}

K = 2


# In[11]:


for u in reviewsPerUser:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(K)]
    
for i in reviewsPerItem:
    itemGamma[i] = [random.random() * 0.1 - 0.05 for k in range(K)]


# Again we must implement an "unpack" function. This is the same as before, though has some additional terms.

# In[12]:


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    global userGamma
    global itemGamma
    index = 0
    alpha = theta[index]
    index += 1
    userBiases = dict(zip(users, theta[index:index+nUsers]))
    index += nUsers
    itemBiases = dict(zip(items, theta[index:index+nItems]))
    index += nItems
    for u in users:
        userGamma[u] = theta[index:index+K]
        index += K
    for i in items:
        itemGamma[i] = theta[index:index+K]
        index += K


# Similarly, our cost and derivative functions serve the same role as before, though their implementations are somewhat more complicated.

# In[13]:


def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item] + inner(userGamma[user], itemGamma[item])


def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['customer_id'], d['product_id']) for d in dataset]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in users:
        cost += lamb*userBiases[u]**2
        for k in range(K):
            cost += lamb*userGamma[u][k]**2
    for i in items:
        cost += lamb*itemBiases[i]**2
        for k in range(K):
            cost += lamb*itemGamma[i][k]**2
    return cost


def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(dataset)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    dUserGamma = {}
    dItemGamma = {}
    for u in reviewsPerUser:
        dUserGamma[u] = [0.0 for k in range(K)]
    for i in reviewsPerItem:
        dItemGamma[i] = [0.0 for k in range(K)]
    for d in dataset:
        u,i = d['customer_id'], d['product_id']
        pred = prediction(u, i)
        diff = pred - d['star_rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
        for k in range(K):
            dUserGamma[u][k] += 2/N*itemGamma[i][k]*diff
            dItemGamma[i][k] += 2/N*userGamma[u][k]*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
        for k in range(K):
            dUserGamma[u][k] += 2*lamb*userGamma[u][k]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
        for k in range(K):
            dItemGamma[i][k] += 2*lamb*itemGamma[i][k]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    for u in users:
        dtheta += dUserGamma[u]
    for i in items:
        dtheta += dItemGamma[i]
    return numpy.array(dtheta)


# Again we optimize using our gradient descent library, and compare to a simple baseline.

# In[14]:


MSE(alwaysPredictMean, labels) #Same as our previous baseline


# In[19]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + # Initialize alpha
                                   [0.0]*(nUsers+nItems) + # Initialize beta
                                   [random.random() * 0.1 - 0.05 for k in range(K*(nUsers+nItems))], # Gamma
                             derivative, args = (labels, 0.001), maxfun = 10, maxiter = 10)

#Note the "maxfun = 10" and "maxiter = 10" this is because this function will go on for over 
# 20 iterations taking far too long to compute.


# Note finally that in the above exercise we only computed the ___training___ error of our model, i.e., we never confirmed that it works well on held-out (validation/testing) data!

# ## You're all done!
# 
# This weeks notebook was fairly simple (homework-wise), but the concepts were rather difficult. Next week you will start your capstone project, which will combine all 4 courses into a single assignment! Remember to use all your available resources when you start the project, including your previous notebooks!
