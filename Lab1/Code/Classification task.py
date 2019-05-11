#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Some imports might be redundant because the code was split up into pieces
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from itertools import groupby
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE


# In[2]:


# Set the plot size
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[3]:


def get_scores(y_predict, y_true):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(y_predict)):
        if y_true[i]==1 and y_predict[i]==1:
            TP += 1
        if y_true[i]==0 and y_predict[i]==1:
            FP += 1
        if y_true[i]==1 and y_predict[i]==0:
            FN += 1
        if y_true[i]==0 and y_predict[i]==0:
            TN += 1
    
    return TP, FP, FN, TN


# In[4]:


# Preprocessing
# Read the data in pandas
data = pd.read_csv("data_for_student_case.csv")
data['bookingdate'] =  pd.to_datetime(data['bookingdate'])
data['creationdate'] =  pd.to_datetime(data['creationdate'])

# Delete the refused transactions (but keep them for later use maybe)

refused_data = data.loc[data['simple_journal'] == "Refused", :]
dataset = data.loc[data['simple_journal'] != "Refused", :].copy()
dataset['bool_fraud'] = (dataset['simple_journal'] == "Chargeback").copy().astype(int)
dataset['bool_valid'] = (dataset['simple_journal'] == "Settled").copy().astype(int)

# Do some preprocessing for the ML algorithms
finalset = dataset.copy()
finalset['mail_id'] = finalset['mail_id'].str.replace('email','')
finalset['ip_id'] = finalset['ip_id'].str.replace('ip','')
finalset['card_id'] = finalset['card_id'].str.replace('card','')


## Transform the data into onehot vectors
targets_for_onehot = ['issuercountrycode', 'txvariantcode','currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode']

new_df = pd.DataFrame([])

for target in targets_for_onehot:
    temp = pd.get_dummies(finalset[target])
    new_df = pd.concat([new_df, temp],axis=1)

new_df = pd.concat([new_df, finalset[['mail_id','ip_id','card_id','bin','amount']]], axis=1)
new_df = new_df.fillna(0)

## Get the features and labels
x = new_df.values
x[x=="NA"] = 0
x = x.astype(float)
y = finalset['bool_fraud'].values


# In[ ]:





# In[ ]:


class WhiteBoxClassifier():
    def __init__(self, columnlists):
        self.cvcsupplied_column = columnlists.index(True) # cvcsupplied = True
        self.cvcresponsecode0_column = columnlists.index(0) # cvcresponsecode = 0


        self.mccredit_column = columnlists.index('mccredit') # mccredit = 1
        self.ecommerce_column = columnlists.index('Ecommerce') # Ecommerce = 1

        self.MX_column = columnlists.index('MX') # MX = 1
         # Ecommerce = 1

        self.AU_column = columnlists.index('AU') # AU = 1
        # Ecommerce = 1
        
        print("Created a classifier")

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass
    
    def apply_single_predict(self, x):
        if (x[self.cvcsupplied_column] == 1) and (x[self.cvcresponsecode0_column] == 1):
            return 1
        
        if (x[self.mccredit_column] == 1) and (x[self.ecommerce_column] == 1):
            return 1
        
        if (x[self.MX_column] == 1) and (x[self.ecommerce_column] == 1):
            return 1
        
        if (x[self.AU_column] == 1) and (x[self.ecommerce_column] == 1):
            return 1
        
        return 0
        
    
    def predict(self, x_test):
        predictions = np.apply_along_axis(self.apply_single_predict, axis=1, arr=x_test)
        return predictions


# In[ ]:


from imblearn.over_sampling import SMOTE

## Create a SMOTE object
sm = SMOTE()


# In[ ]:


## Do the K-fold crossvalidation
w_clf = WhiteBoxClassifier(new_df.columns.tolist())

total_TP_W = 0
total_FP_W = 0
total_FN_W = 0
total_TN_W = 0

total_TP_B = 0
total_FP_B = 0
total_FN_B = 0
total_TN_B = 0

k_fold = KFold(n_splits=10)
iteration = 0
for train_indices, test_indices in k_fold.split(x):
    iteration = iteration +1
    print("Iteration " + str(iteration))
    x_train = x[train_indices,:]
    y_train = y[train_indices]
    x_test = x[test_indices,:]
    y_test = y[test_indices]
    
    predictions = w_clf.predict(x_test)
    TP_W, FP_W, FN_W, TN_W = get_scores(predictions, y_test)
    
    total_TP_W = total_TP_W + TP_W
    total_FP_W = total_FP_W + FP_W
    total_FN_W = total_FN_W + FN_W
    total_TN_W = total_TN_W + TN_W

    b_clf = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
    predictions = b_clf.predict(x_test)
    TP_B, FP_B, FN_B, TN_B = get_scores(predictions, y_test)
    
    total_TP_B = total_TP_B + TP_B
    total_FP_B = total_FP_B + FP_B
    total_FN_B = total_FN_B + FN_B
    total_TN_B = total_TN_B + TN_B
    


# In[ ]:


print("Whitebox classifier results:")
print("TP Whitebox:\t" + str(total_TP_W))
print("FP Whitebox:\t" + str(total_FP_W))
print("FN Whitebox:\t" + str(total_FN_W))
print("TN Whitebox:\t" + str(total_TN_W))
print()

print("Blackbox classifier results:")
print("TP Blackbox:\t" + str(total_TP_B))
print("FP Blackbox:\t" + str(total_FP_B))
print("FN Blackbox:\t" + str(total_FN_B))
print("TN Blackbox:\t" + str(total_TN_B))
print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




