#!/usr/bin/env python
# coding: utf-8

# # Decision Tree classifier and its visualization  

# # Shivani Agrawal

# In[4]:


#importing libraries
import pandas as pd
import numpy as np
from sklearn import tree


# In[5]:


data=pd.read_csv(r'C:\Users\Admin\Downloads\Iris.csv')
data.head()                 


# In[6]:


inputs=data.drop(['Species','Id'],axis=1)
target=data['Species']


# In[7]:


inputs,target


# # Splitting the Data

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


inputs_train,inputs_test,target_train,target_test=train_test_split(inputs,target,test_size=0.2,random_state=42)


# In[10]:


model=tree.DecisionTreeClassifier(max_depth=10)
model.fit(inputs_train,target_train)


# In[11]:


model.fit(inputs, target).get_params()


# In[12]:


target_pred = model.predict(inputs_test)
target_pred


# In[13]:


from sklearn import metrics
metrics.accuracy_score(target_test,target_pred)


# # Visualization

# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize=(28,20))
fig= tree.plot_tree (model , feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],
                   class_names=['Iris-versicolor', 'Iris-setosa', 'Iris-virginica' ],
                   filled=True)

