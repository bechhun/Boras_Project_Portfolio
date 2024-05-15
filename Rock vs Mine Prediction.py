#!/usr/bin/env python
# coding: utf-8

# ### Importing Data

# In[57]:


import pandas as pd
import csv
csv = r'C:\Users\u122398\OneDrive - Straumann Group\Desktop\Copy of sonar data.csv'
df = pd.read_csv(csv, sep=',', header=None)
df


# ### Importing dependencies

# In[58]:


import numpy as np ##Used for creating numpy arrays
import pandas as pd ##Used to load our data into tables called dataframes
from sklearn.model_selection import train_test_split ## The library is sklearn
from sklearn.linear_model import LogisticRegression ##The model we will use for this project
from sklearn.metrics import accuracy_score ##Used to check the accuracy of our prediction


# ### Data Collection and Data Processing

# In[59]:


df.head()


# In[60]:


# number of rows and columns
df.shape


# In[61]:


# Gives some statistical analysis - Count, Mean, STD, Min, percentile, Max for the columns
df.describe()

# count gives you the number os rows for that column. 
# Mean gives you the mean or avg for that column.
# std gives you the standard deviation from the mean.
# 25% means 25 percent of the data is lower than the number.
# 50% means 50 percent of the data is lower than the number.
# 75% means 75 percent of the data is lower than the number.
# Max means whats the max number of the column.


# In[62]:


df[60].value_counts() # <-- This fuctions counts the difference values for the column.
                      #[60] <-- the column index. 


# M = Mine
# 
# R = Rock

# In[63]:


df.groupby(60).mean()


# ### Seperating Data and labels

# In[64]:


# We need to seperate the data to do prediction. 
X = df.drop(columns = 60, axis=1) #<-- This stores the data frame as X but does not include the last column(Rock or mine column)
Y = df[60] #<-- this means create a dataframe call Y with only the 60th column from df. 


# In the code X = df.drop(columns=60, axis=1), you're using pandas to drop a specific column from a DataFrame (df). The axis=1 parameter indicates that you want to drop a column, rather than a row.
# To make it simpler, imagine you have a big table of data, like a spreadsheet. Each row might represent a different observation or sample, and each column represents a different feature or characteristic.
# #So, when you set axis=1, you're saying "I want to drop something along the columns axis," which means you're removing a column from your table of data. In this case, you're dropping the column labeled 60.
# After executing this line of code, X will contain the DataFrame df with the column labeled 60 removed. It's like taking a pair of scissors and cutting out that specific column from your table of data.

# In[36]:


print(X) #<-- Prints the new table that we name X (You can still call df and it'll give you the original df table)
print(Y) #<-- This will give you only the last column which is the rock or mine column. 


# ### Splitting Data to Training and Test Data
# #### Why do we split training and test data?
# 
# Okay, imagine you're learning to bake cookies. You have your ingredients, and you want to make sure your recipe is just right. Now, instead of baking all your cookies at once, you decide to split your dough into two parts.
# 
# One part, let's call it the "practice dough," you use to try out your recipe and get better at baking. You mix it, shape it, and bake it, making sure everything turns out yummy.
# 
# The other part, we'll call it the "test dough," you set aside. You don't touch it until you're completely done practicing. Once you're confident in your baking skills, you take out the test dough and bake it. This way, you can see if your cookies turn out as tasty as you expect.
# 
# Splitting data into training and test data in computer stuff, like machine learning, works the same way. The training data is like your practice dough. You use it to teach your computer model how to do things, like recognizing pictures of cats or predicting the weather.
# 
# But just like with baking, you want to make sure your model isn't just good at remembering the training data. So, you set aside some test data that your model hasn't seen before. When your model is all trained up, you test it on this data to see if it's really learned what you wanted it to.
# 
# It's like making sure your cookie recipe works not just for the cookies you practiced with, but for any cookie dough you might bake in the future!

# In[37]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1 )

# We are splitting the X and Y data set into training and test data.
# test_size = 0.1 means that 10% of the data in X and Y data set needs to be test data. (0.1 = 10%, 0.2 = 20% and so on.)
# Stratify = Y means to mean split the data based on rock and mine.
# random_state = 1 means to split the data in a particular order. 


# In[38]:


print(X.shape, X_train.shape, X_test.shape)


# In[40]:


print(X_train)
print(Y_train)


# ### Model Training --> Logistic Regression 

# In[44]:


model = LogisticRegression()


# In[45]:


#training the logistic Regression model with training data
model.fit(X_train, Y_train)


# ### Model Evaluation

# accuracy of the model - Most of the time the accuracy on the training data will be better than the test data set because the model already knows the training data set. It's like playing a video game in the same setting over and over again. You already know the setting so nothing really changes and your skill stays the same for it until you go into a new setting thats where things become more challenging. 

# In[46]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# We are naming the X_train prediction output as X_train_prediction.
# We are naming the accuracy score of the model training_data_accuracy
# 
# model.predict is the function and X_train is the parameter
# accuracy_score is the function and X_train_prediction and Y_train is the parameter. 

# In[47]:


print('Accuracy on training data : ', training_data_accuracy)


# In[48]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[49]:


print('Accuracy on training data : ', test_data_accuracy)


# Same as the training set but with the test data sets.

# ### Making a predictive System

# In[56]:


input_data = (0.0261,0.0266,0.0223,0.0749,0.1364,0.1513,0.1316,0.1654,0.1864,0.2013,0.2890,0.3650,0.3510,0.3495,0.4325,0.5398,0.6237,0.6876,0.7329,0.8107,0.8396,0.8632,0.8747,0.9607,0.9716,0.9121,0.8576,0.8798,0.7720,0.5711,0.4264,0.2860,0.3114,0.2066,0.1165,0.0185,0.1302,0.2480,0.1637,0.1103,0.2144,0.2033,0.1887,0.1370,0.1376,0.0307,0.0373,0.0606,0.0399,0.0169,0.0135,0.0222,0.0175,0.0127,0.0022,0.0124,0.0054,0.0021,0.0028,0.0023)

# changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape as numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction= model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




