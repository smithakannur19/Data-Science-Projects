#!/usr/bin/env python
# coding: utf-8

# # Week 1: Similarity-Based Recommendation Systems
# 
# In course 3 we learned about classification and methods of supervised learning. In ths course we will discuss the use of a recommendation system and give a recap of this series of courses and what we have learned.

# ## Part 1: Setting up the Data
# 
# This dataset is a series of reviews and ratings of Digital Video Games from Amazon. 
# https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz

# In[1]:


import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import numpy
import random

path = "/Users/Cheedo/Documents/wMF/amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz"


# We are going to set up our data in the same way we did in Course 3. Based on Ratings and Votes we will make recomendations based on a selected Game.

# In[2]:


f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

dataset = []

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)


# Let's look at what a typical entry will look like in this dataset.

# In[3]:


dataset[0]


# ## Part 2: Finding Similarities
# 
# In Course 3 we learned how to take the above review and predict a star rating (or any other value) by using models which gave each word in a review a weight and predicted the rating based on the sum of those weights. Now we will learn the basic ideas behind how to make a Recommendation. The parts of our data we want to work with are "Star Rating", "HelpFul Votes", and "Total Votes."

# In[4]:


usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

itemNames = {}

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']


# ### Functions to find Similarities
# 
# We need to set up our Jaccard function and a function to determine what is similar within the dataset.

# In[5]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[6]:


def mostSimilar(iD, n):
    similarities = []
    users = usersPerItem[iD]
    for i2 in usersPerItem:
        if i2 == iD: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:n]


# ### Getting a recommendation
# 
# Our mostSimilar function above takes an input of a "Product ID" and a value n which is the number of similar Products we would like, then outputs a list of size n with the Poduct ID's.

# In[7]:


dataset[2]


# In[8]:


query = dataset[10]['product_id']
query


# In[9]:


itemNames[query]


# In[10]:


mostSimilar(query, 10)


# In[11]:


#TODO Get a list of 10 most similar product names to our query defined above

### Note we want PRODUCT NAMES here, not ID's

[itemNames[x["TODO"]] for x in mostSimilar(query, 10)]


# Notice that in this example, we get 3 different instances of "Sid Meier's Civilization V". We aren't taking into account the fact that multiple digital games share the same beginning characters since they have extra downloadable content for the same game. This is good in basic recommendation systems as someone who bought the original query of "Sid Meier's Civilization V" is likely to be interested in extra content for the game.

# ## Part 3: Collaborative-Filtering-Based Rating Estimation
# 
# We can also use the similarity-based recommender we developed above to make predictions about user's ratings. Although this is not an example of machine learning, it is a simple heuristic that can be used to estimate a user's future ratings based on their ratings in the past.
# 
# Specifically, a user's rating for an item is assumed to be a weighted sum of their previous ratings, weighted by how similar the query item is to each of their previous purchases.
# 
# We start by building a few more utility data structures to keep track of all of the reviews by each user and for each item.

# In[12]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)


# In[13]:


#TODO Calculate the mean rating of the entire dataset

#Answer should be roughly 3.853...


# Now that we have calculated the average rating of our dataset as a whole, we are going to implement a function which predicts Rating based on a user and an item.

# In[14]:


def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(d['star_rating'])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[15]:


dataset[10]


# In[25]:


#TODO Using the function defined above, calculate the predicted rating for the user at index [10]

user,item = dataset[ "TODO" ]['customer_id'], dataset[ "TODO" ]['product_id']
predictRating(user, item)


# In this case our user hasn't rated any similar items, so our function defaults to returning the dataset Mean Rating. Let's try another example with a user who has.

# In[26]:


#TODO Calculate the predicted rating for the user at index [12]

#Answer should differ from the above


# ## Part 4: Evaluating Performance
# 
# Lets start by defining out typical MSE function.

# In[27]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# To evaluate the performance of our model, we will need two things:
# 1. A list of the average Rating (i.e. ratingMean)
# 2. A list of our predicted ratings (i.e. ratings defined by our predictRating function)

# In[29]:


#TODO Define the two lists described above


# Finally, we will compare our two lists above with the actual star ratings in our dataset.

# In[30]:


labels = [d['star_rating'] for d in dataset]

print(MSE(alwaysPredictMean, labels), MSE(cfPredictions, labels))


# In this case, the accuracy of our rating prediction model was _nearly identical_ (in terms of the MSE) than just predicting the mean rating. However note again that this is just a heuristic example, and the similarity function could be modified to change its predictions (e.g. by using a different function other than the Jaccard similarity).

# ## You're all done!
# 
# This week was an introduction to the basics of recomender systems. Next week we will go over Latent Factor Models and how to implement them.
