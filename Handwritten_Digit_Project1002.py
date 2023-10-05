#!/usr/bin/env python
# coding: utf-8

# # Number Recognition

# Handwritten digit recognition system not only detects
# scanned images of handwritten digits.Handwritten digit
# recognition using MNIST dataset is a major project made

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


len(X_test)


# In[5]:


X_train[0].shape


# In[6]:


X_train[0]


# In[7]:


plt.matshow(X_train[0])


# In[8]:


y_train[0]


# In[9]:


X_train = X_train / 255
X_test = X_test / 255


# In[10]:


X_train[0]


# In[11]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[12]:


X_train_flattened.shape


# In[13]:


X_train_flattened[0]


# In[14]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)


# In[15]:


model.evaluate(X_test_flattened, y_test)


# In[16]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[17]:


plt.matshow(X_test[0])


# In[18]:


np.argmax(y_predicted[0])


# In[19]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[20]:


y_predicted_labels[:5]


# In[21]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[22]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# **Using hidden layer**

# In[23]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)


# In[24]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# **Using Flatten layer so that we don't have to call .reshape on input dataset**

# In[25]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[26]:


model.evaluate(X_test,y_test)

