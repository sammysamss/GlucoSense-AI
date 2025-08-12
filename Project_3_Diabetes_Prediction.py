#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection and Analysis
# 
# PIMA Diabetes Dataset

# In[ ]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[ ]:


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[ ]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[ ]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[ ]:


diabetes_dataset['Outcome'].value_counts()


# 0 --> Non-Diabetic
# 
# 1 --> Diabetic

# In[ ]:


diabetes_dataset.groupby('Outcome').mean()


# In[ ]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[ ]:


print(X)


# In[ ]:


print(Y)


# Data Standardization

# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(X)


# In[ ]:


standardized_data = scaler.transform(X)


# In[ ]:


print(standardized_data)


# In[ ]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[ ]:


print(X)
print(Y)


# In[ ]:





# Train Test Split

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

# In[ ]:


classifier = svm.SVC(kernel='linear')


# In[ ]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[ ]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[ ]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[ ]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[ ]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Making a Predictive System

# In[ ]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




