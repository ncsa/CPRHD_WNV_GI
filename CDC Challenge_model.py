#!/usr/bin/env python
# coding: utf-8

# In[364]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import GRU
from keras import regularizers
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sn
import itertools  
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import random
from sklearn.utils import resample


# In[605]:


#Function to print confusion matrix in a nice format.
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


# In[606]:


##Specifying training data parameters.
#years=range(2000,2019)
years=range(2000,2010)
num_years=2
#Features/columns that will be used for prediction.
features=["count","neighborCountyAvg", "Gini", "Temp", "Prec", "Hum", "County_type", "Resident_population_White_alone_percent", "Median_Household_Income", "Poverty_percent_of_people"]
#features=["count","neighborCountyAvg", "Gini", "Temp", "Prec", "Hum", "County_type"]


# In[585]:


#Importing training and testing data
train_path="/home/sparsha2/data/Temporal_df.csv"
temporal_df=pd.read_csv(train_path)
test_path="/home/sparsha2/data/Temporal_df_test.csv"
temporal_df_test=pd.read_csv(test_path)


# In[586]:


temporal_df.head()


# In[607]:


#Checking frequency of instances for each class
dict_freq={}
for i in range(1,16):
    dict_freq[i]=len(temporal_df[temporal_df[str(i)]==1])
dict_freq


# In[608]:


#Oversampling to balance the classes
num_cl=len(temporal_df[temporal_df["1"]==1])
df_balanced=temporal_df[temporal_df["1"]==1]
for i in range(2,16):
    df_temp=temporal_df[temporal_df[str(i)]==1]
    if(len(df_temp)>0):
        df_minority_upsampled = resample(df_temp, replace=True, n_samples=num_cl, random_state=4)
        df_balanced=pd.concat([df_balanced, df_minority_upsampled])
df_balanced = df_balanced.sample(frac = 1, random_state=4)   #Shuffling the data
X_pre = df_balanced.iloc[:, 0:(num_years*len(features))]
Y_pre = df_balanced.iloc[:, (num_years*len(features)):]


# In[609]:


# #Using unbalanced data
# temporal_df = temporal_df.sample(frac = 1) #Shuffling the data
# X_pre = temporal_df.iloc[:, 0:(num_years*len(features))]
# Y_pre = temporal_df.iloc[:, (num_years*len(features)):]


# In[610]:


X=X_pre.values
Y=Y_pre.values


# In[611]:


# #Balancing the data w.r.t. class labels
# ros = RandomOverSampler()
# X, Y = ros.fit_resample(X_pre.values,Y_pre.values)


# In[612]:


#Shows all the class labels are now equally represented
y=np.argmax(Y, axis=1)
np.bincount(y)


# In[564]:


# # Calculating classs weights based on their frequencies
# y=np.argmax(temporal_df.iloc[:,-15:].values, axis=1)
# class_weights=compute_class_weight("balanced", [i for i in range(15)], y)


# ## LSTM Implementation for classification

# In[494]:


def create_network():
    network = Sequential()
    network.add(BatchNormalization(input_shape=(num_years, len(features))))
    network.add(Dense(4, activation="tanh"))
    network.add(LSTM(15, dropout = 0.2, recurrent_dropout = 0.2, activation="tanh"))
    network.add(Dense(15, activation="softmax"))
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return network


# In[495]:


#Cross-validation setup
acc_train=[]
acc_val=[]
kf = KFold(n_splits=5, shuffle=True)
for train_index, val_index in kf.split(X):   
    X_train=X[train_index]
    Y_train=Y[train_index]
    X_val=X[val_index]
    Y_val=Y[val_index]
    
    #Transforming input variables into LSTM input format
    X_train = X_train.reshape(X_train.shape[0], num_years, len(features))
    X_val = X_val.reshape(X_val.shape[0], num_years, len(features))
    Y_train=Y_train
    Y_val=Y_val
    
    #Creating model
    model=create_network()
    Hist=model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), verbose=2, class_weight=None)
    
    #Final epoch accuracies for training and validation dataset
    acc_train.append(Hist.history["accuracy"][-1])
    acc_val.append(Hist.history["val_accuracy"][-1])
    
print("Training accuracy:" + str(np.mean(acc_train)))
print("Validation accuracy:" + str(np.mean(acc_val)))
    


# In[593]:


# Dataset while using test_train split for final prediction

#Transforming input variables into LSTM input format

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20 , random_state=4)

X_train = X_train.reshape(X_train.shape[0], num_years, len(features))
X_val = X_val.reshape(X_val.shape[0], num_years, len(features))
Y_train=Y_train
Y_val=Y_val

X_test=temporal_df_test.iloc[:, 0:(num_years*len(features))]
X_test = X_test.values.reshape(X_test.shape[0], num_years, len(features))


# In[594]:


model=create_network()


# In[595]:


Hist=model.fit(X_train, Y_train, nb_epoch=40, validation_data=(X_val, Y_val), verbose=2, class_weight=None)


# In[596]:


# Plot training & validation accuracy values
plt.plot(Hist.history['accuracy'])
plt.plot(Hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(Hist.history['loss'])
plt.plot(Hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[597]:


# #Using unbalanced data
# temporal_df = temporal_df.sample(frac = 1) #Shuffling the data
# X_pre = temporal_df.iloc[:, 0:(num_years*len(features))]
# Y_pre = temporal_df.iloc[:, (num_years*len(features)):]
# X=X_pre.values
# Y=Y_pre.values
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20)

# X_train = X_train.reshape(X_train.shape[0], num_years, len(features))
# X_val = X_val.reshape(X_val.shape[0], num_years, len(features))
# Y_train=Y_train
# Y_val=Y_val


# In[613]:


#Predicting value for train, val, and test datasets
pred_train=model.predict(X_train)
pred_val=model.predict(X_val)
#pred_test=model.predict(X_test)


# In[614]:


#Converting probabilities to class labels
pred_train_class=np.argmax(pred_train, axis=1)+1
pred_train_class=list(map(lambda x: str(x), pred_train_class))
pred_val_class=np.argmax(pred_val, axis=1)+1
pred_val_class=list(map(lambda x: str(x), pred_val_class))
#pred_test_class=np.argmax(pred_test, axis=1)+1


# In[615]:


true_train_class=np.argmax(Y_train, axis=1)+1
true_train_class=list(map(lambda x: str(x), true_train_class))
true_val_class=np.argmax(Y_val, axis=1)+1
true_val_class=list(map(lambda x: str(x), true_val_class))


# In[616]:


labels = [str(i) for i in range(1,16)]
cm_train = confusion_matrix(true_train_class, pred_train_class , labels)
#cm_train=cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] #For normalizing
cm_val = confusion_matrix(true_val_class, pred_val_class , labels)
#cm_val=cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis] #For normalizing


# In[617]:


print_cm(cm_train, labels)


# In[618]:


print_cm(cm_val, labels)


# In[619]:


#Evaluation metrics for valdation dataset
print(metrics.classification_report(true_val_class, pred_val_class))


# In[ ]:




