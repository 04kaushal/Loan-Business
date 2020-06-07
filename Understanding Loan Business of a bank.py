#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import warnings
#import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import time
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\ezkiska\Videos\Imarticus\Python\7th Week project 18th and 19th\SAT project")

df1 = pd.read_csv(r'borrower_table.csv')
df2 = pd.read_csv(r'loan_table.csv')


# In[2]:


df1.head()


# In[3]:


df2.head()


# In[4]:


df1.columns


# In[5]:


df2.columns


# In[6]:


data=pd.merge(df1, df2, on='loan_id') #default inner
data.columns


# In[7]:


data.describe()


# In[8]:


data.describe(include = 'O')


# In[9]:


data.dtypes


# In[10]:


data.isnull().sum() 

'' #fully repaid & currently repaying has 54947 
NAN observations, Avg percentage has 6972 & load repaid has 53446 NAN
'''
# LOAD Repaid is target variable
#Test NAN on Load_repaid as these have not taken load
#load granted is a better option


'''drop load granted & ID from train test

'egregetting Columns based on Data types '''
# In[11]:


data.drop(['loan_id'], axis=1, inplace=True)
data.dtypes


# In[59]:


data['loan_repaid'].value_counts()


# In[12]:


df_nom=data.select_dtypes(include='object')
df_ordinal = data [['is_first_loan','is_employed','fully_repaid_previous_loans','currently_repaying_other_loans',
                     'avg_percentage_credit_card_limit_used_last_year','loan_granted','loan_repaid']]
df_numeric = data [['total_credit_card_limit','saving_amount','checking_amount','yearly_salary','age',
                     'dependent_number']]


# #### Working on ordinal data 

# In[13]:


df_ordinal.isnull().sum() 


# In[14]:


df_ordinal.columns


# In[15]:


df_ordinal[['fully_repaid_previous_loans','currently_repaying_other_loans']]=df_ordinal[['fully_repaid_previous_loans',
          'currently_repaying_other_loans']].fillna(value=2)
df_ordinal['avg_percentage_credit_card_limit_used_last_year'] = df_ordinal['avg_percentage_credit_card_limit_used_last_year'].fillna(axis =0, 
          method ='ffill')
sns.distplot(df_ordinal['avg_percentage_credit_card_limit_used_last_year'])


# In[16]:


df_ordinal['avg_percentage_credit_card_limit_used_last_year'].isnull().sum()


# In[17]:


df_ordinal[['avg_percentage_credit_card_limit_used_last_year', 'loan_repaid']].groupby(['loan_repaid'], 
          as_index=False).mean().sort_values(by='loan_repaid', ascending=True)
'''since valuers are very close by we will have to keep it wather that scalling to 0 or 1'''


# In[18]:


df_numeric['avg_percentage_credit_card_limit_used_last_year'] = df_ordinal['avg_percentage_credit_card_limit_used_last_year']
df_ordinal.drop(['avg_percentage_credit_card_limit_used_last_year'], axis=1, inplace=True)
'''
Now Ordinal is donw, we will work on df_numeric we will scale & do quantile cut'''


# In[19]:


from sklearn import preprocessing

## scaling
scaler = preprocessing.MinMaxScaler().fit(df_numeric)
scaled_numeric = scaler.transform(df_numeric)
dfs_numeric=pd.DataFrame(scaled_numeric, columns=df_numeric.columns.tolist())


# In[20]:


'''now quantile cut on pd.numeric original data '''

df_numeric['total_credit_card_limit_band'] = pd.qcut(df_numeric['total_credit_card_limit'],4)#.value_counts()
df_numeric['saving_amount_band'] = pd.qcut(df_numeric['saving_amount'],4)#.value_counts()
df_numeric['checking_amount_band'] = pd.qcut(df_numeric['checking_amount'],4)#value_counts()
df_numeric['age_band'] = pd.qcut(df_numeric['age'],4)#.value_counts() #df_numeric['age'] check Q cut values
df_numeric['avg_percentage_credit_card_limit_used_last_year_band'] = pd.qcut(df_numeric['avg_percentage_credit_card_limit_used_last_year'],4)#.value_counts()
df_numeric['dependent_number_band'] = pd.qcut(df_numeric['dependent_number'],4)


# In[21]:


#pd.qcut(df_numeric['yearly_salary'],3) #giving error Unexpected duplicates each array should  contain unique 1st value
nonZeroRows = df_numeric.loc[df_numeric['yearly_salary']!=0, 'yearly_salary']
pd.qcut(nonZeroRows,4).value_counts()


# In[22]:


# Quantile cut on 
df_numeric.loc[(df_numeric['yearly_salary']>99.999) & (df_numeric['yearly_salary']<= 21900.0 ),'yearly_salary'] = 1
df_numeric.loc[(df_numeric['yearly_salary']>21900.0) & (df_numeric['yearly_salary']<= 30900.0 ),'yearly_salary'] = 2
df_numeric.loc[(df_numeric['yearly_salary']>30900.0) & (df_numeric['yearly_salary']<= 41100.0 ),'yearly_salary'] = 3
df_numeric.loc[(df_numeric['yearly_salary']>41100.0) & (df_numeric['yearly_salary']<= 97200.0 ),'yearly_salary'] = 4
# check 
df_numeric['yearly_salary'].value_counts()


# In[23]:


# Quantile cut on  total_credit_card_limit
df_numeric.loc[(df_numeric['total_credit_card_limit']>0.0) & (df_numeric['total_credit_card_limit']<= 2700.0 ),'total_credit_card_limit'] = 0
df_numeric.loc[(df_numeric['total_credit_card_limit']>2700.0) & (df_numeric['total_credit_card_limit']<= 4100.0 ),'total_credit_card_limit'] = 1
df_numeric.loc[(df_numeric['total_credit_card_limit']>4100.0) & (df_numeric['total_credit_card_limit']<= 5500.0 ),'total_credit_card_limit'] = 2
df_numeric.loc[(df_numeric['total_credit_card_limit']>5500.0) & (df_numeric['total_credit_card_limit']<= 13500.0 ),'total_credit_card_limit'] = 3

# Quantile cut on  saving_amount
df_numeric.loc[(df_numeric['saving_amount']>0.0) & (df_numeric['saving_amount']<= 834.0 ),'saving_amount'] = 0
df_numeric.loc[(df_numeric['saving_amount']>834.0) & (df_numeric['saving_amount']<= 1339.0 ),'saving_amount'] = 1
df_numeric.loc[(df_numeric['saving_amount']>1339.0) & (df_numeric['saving_amount']<= 2409.0 ),'saving_amount'] = 2
df_numeric.loc[(df_numeric['saving_amount']>2409.0) & (df_numeric['saving_amount']<= 10641.0 ),'saving_amount'] = 3


# Quantile cut on  checking_amount
df_numeric.loc[(df_numeric['checking_amount']>0.0) & (df_numeric['checking_amount']<= 1706.0 ),'checking_amount'] = 0
df_numeric.loc[(df_numeric['checking_amount']>1706.0) & (df_numeric['checking_amount']<= 2673.0 ),'checking_amount'] = 1
df_numeric.loc[(df_numeric['checking_amount']>2673.0) & (df_numeric['checking_amount']<= 4241.0 ),'checking_amount'] = 2
df_numeric.loc[(df_numeric['checking_amount']>4241.0) & (df_numeric['checking_amount']<= 13906.0 ),'checking_amount'] = 3

# Quantile cut on  yearly_salary
df_numeric.loc[(df_numeric['yearly_salary']>0.0) & (df_numeric['yearly_salary']<= 18900.0 ),'yearly_salary'] = 0
df_numeric.loc[(df_numeric['yearly_salary']>18900.0) & (df_numeric['yearly_salary']<= 29400.0 ),'yearly_salary'] = 1
df_numeric.loc[(df_numeric['yearly_salary']>29400.0) & (df_numeric['yearly_salary']<= 40200.0 ),'yearly_salary'] = 2
df_numeric.loc[(df_numeric['yearly_salary']>40200.0) & (df_numeric['yearly_salary']<= 97200.0 ),'yearly_salary'] = 3


# In[24]:


# Quantile cut on  age

'''
df_numeric.age.min()
Out[69]: 18
df_numeric.age.max()
Out[70]: 79

33- 47 -> 1200- 1405
-32-> 1198

'''
df_numeric.loc[(df_numeric['age']>17.999) & (df_numeric['age']<= 32.0 ),'age'] = 0
df_numeric.loc[(df_numeric['age']>32.0) & (df_numeric['age']<= 41.0 ),'age'] = 1
df_numeric.loc[(df_numeric['age']>41.0) & (df_numeric['age']<= 50.0 ),'age'] = 2
df_numeric.loc[(df_numeric['age']>50.0) & (df_numeric['age']<= 79.0 ),'age'] = 3


# In[25]:


# Quantile cut on  avg_percentage_credit_card_limit_used_last_year_band
df_numeric.loc[(df_numeric['avg_percentage_credit_card_limit_used_last_year']>0.00) & (df_numeric['avg_percentage_credit_card_limit_used_last_year']<= 0.6 ),'avg_percentage_credit_card_limit_used_last_year'] = 0
df_numeric.loc[(df_numeric['avg_percentage_credit_card_limit_used_last_year']>0.6) & (df_numeric['avg_percentage_credit_card_limit_used_last_year']<= 0.73 ),'avg_percentage_credit_card_limit_used_last_year'] = 1
df_numeric.loc[(df_numeric['avg_percentage_credit_card_limit_used_last_year']>0.73) & (df_numeric['avg_percentage_credit_card_limit_used_last_year']<= 0.86 ),'avg_percentage_credit_card_limit_used_last_year'] = 2
df_numeric.loc[(df_numeric['avg_percentage_credit_card_limit_used_last_year']>0.86) & (df_numeric['avg_percentage_credit_card_limit_used_last_year']<= 1.09 ),'avg_percentage_credit_card_limit_used_last_year'] = 3

#dependent_number
df_numeric.loc[(df_numeric['dependent_number']>17.999) & (df_numeric['dependent_number']<= 2.0 ),'dependent_number'] = 0
df_numeric.loc[(df_numeric['dependent_number']>2.0) & (df_numeric['dependent_number']<= 3.0 ),'dependent_number'] = 1
df_numeric.loc[(df_numeric['dependent_number']>3.0) & (df_numeric['dependent_number']<= 6.0 ),'dependent_number'] = 2
df_numeric.loc[(df_numeric['dependent_number']>6.0) & (df_numeric['dependent_number']<= 8.0 ),'dependent_number'] = 3

df_numeric.drop(df_numeric.iloc[:, 7:], inplace = True, axis = 1)


# In[26]:


'''Re[eat for all]


Dumyfying for Nom data , 1st drop date
'''

df_nom.drop(['date'], axis=1, inplace=True)

df_nom_dum = pd.get_dummies(df_nom, drop_first = True)

'''
Now impute in Nom

'''

from sklearn.preprocessing import LabelEncoder
lbs = LabelEncoder()
lbs.fit(df_nom)
df_nom_labels = lbs.transform(df_nom)
df_nom['loan_purpose'] = df_nom_labels


# In[27]:


'''
#ordinal & nominal will be same for all model
#Log reg , KNN, SVM will use Scaled numeric, labbel & dummiees for nom will both be used
#DT XG, RF wil use Qcur numerical


Do SMOTE on train
Split in train and test
1. Cross validation - Grid search mmodel name , cross folds
2. 


'''
df3 = pd.concat([df_nom_dum, df_numeric, df_ordinal], axis = 1) # dummyfying nominal + Qcut numerical

train=df3[df3['loan_granted']==1]
test=df3[df3['loan_granted']==0]

len(train[train['loan_repaid']==1])/train.shape[0]

train = train.drop('loan_granted', axis = 1)
test = test.drop('loan_granted', axis = 1)

dataforDTreewithouttSMOTE = train


# ### ----------------------------------- MODELLING-----------------------------------------------------------------------------
# 

# In[28]:


from imblearn.over_sampling import SMOTE #reduces oversampling/ technique
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report , f1_score 
from sklearn.tree import DecisionTreeClassifier
from confusionMatrix import plotConfusionMatrix

ywithoutsmote = dataforDTreewithouttSMOTE['loan_repaid']
Xwithoutsmote = dataforDTreewithouttSMOTE.iloc[:,0:-1]


# #####  Method 0: without SMOTE---------------------------------------

# In[29]:


X_train, X_val, y_train, y_val = train_test_split(Xwithoutsmote, ywithoutsmote, test_size = 0.3, random_state = 0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf.score(X_train, y_train)
clf.score(X_val, y_val)

predictions_ = clf.predict(X_val) 
  
# print classification report 
print('Without imbalance treatment:'.upper())
print(classification_report(y_val, predictions_)) 
print(confusion_matrix(y_val, predictions_))
f1= f1_score(y_val,predictions_, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val == 0)
sum(y_val == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)


# #####  Method1: SMOTE on Train------------------------------------------------------------------

# In[30]:


sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X_train, y_train.ravel()) 

print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_res.shape)) 
print('After OverSampling, y: {}'.format(y_res.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_res == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_res == 0))) 
print('\n')

#split into 70:30 ratio 
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_res, y_res, 
                    test_size = 0.3, random_state = 0)


# In[31]:


#Checking on Validation Set
clf.fit(X_train_res, y_train_res.ravel()) 
clf.score(X_train_res, y_train_res)
clf.score(X_val_res, y_val_res)

predictions = clf.predict(X_val_res) 
  
# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_res, predictions)) 
print(confusion_matrix(y_val_res, predictions))
f1= f1_score(y_val_res,predictions, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val_res,predictions)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val_res == 0)
sum(y_val_res == 1)
sum(predictions == 0)
sum(predictions == 1)

confusion_matrix(y_val_res, predictions)


# In[34]:


#Cheking on test data :  X_val, y_val

clf.fit(X_train_res, y_train_res.ravel()) 
clf.score(X_val, y_val)

predictions = clf.predict(X_val) 
  
# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val, predictions)) 
print(confusion_matrix(y_val, predictions))
f1= f1_score(y_val,predictions, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val,predictions)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val == 0)
sum(y_val == 1)
sum(predictions == 0)
sum(predictions == 1)

confusion_matrix(y_val, predictions)


# #### GRID SEARCH WITH SMOTE---------------------------------------------------

# In[42]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}   X_val, y_val
parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_res,y_train_res)
preds = tree.predict(X_val)
accu = accuracy_score(y_val, preds)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Accuracy:', np.round(accu,3))
print('*'*80)
print('Using best parameters:',tree.best_params_)
print('*'*80)
print(classification_report(y_val, preds))
print('*'*80)
print(confusion_matrix(y_val, preds))
print('*'*80)
y_pred_proba_ = tree.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba_)
auc = roc_auc_score(y_val, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

print('*'*80)


# #### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------

# In[43]:



def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(5, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_train_res,y_train_res, 4)
acc = model.score(X_val, y_val)
print('Using best parameters:',best_param)
print('accuracy:', np.round(acc,3))

## ROC curve
y_pred_proba = model.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba)
auc = metrics.roc_auc_score(y_val, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()


# #### ---------- Random Forest----------------------------------------

# In[46]:



from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [100,300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier()
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    predict_gscv=rf_gscv.predict(X)
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf, rf_gscv.best_params_, rf_gscv, predict_gscv, score_gscv  

print('*'*80)
print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, pred_rf, acc_rf = RF_grid_search(X_train_res,y_train_res, 4)
pred_rf = model_rf.predict(X_val)
acc_rf = model_rf.score(X_val, y_val)
print('Using best parameters:',best_param_rf)
print('accuracy with Gs:', np.round(acc_rf,3))
print('*'*80)
print(classification_report(y_val, pred_rf))
print('*'*80)
print(confusion_matrix(y_val, pred_rf))
print('*'*80)

## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_val)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_val,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_val, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ## XG Boost ::

# In[48]:


# import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train_res, y_train_res)

# make predictions for test set
y_pred = model_G.predict(X_val)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_val,y_val)

print('*'*80)
print('SMOTE with XGB:')
print("Accuracy without Gs: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_val, y_pred))
print('*'*80)
print(confusion_matrix(y_val, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_val)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_val, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')


# ### ''' Now changing the combination i.e. taking Label encoding nominal'''

# In[49]:


df4 = pd.concat([df_nom, df_numeric, df_ordinal], axis = 1) # label nominal + Qcut numerica
train1=df4[df4['loan_granted']==1]
test1=df4[df4['loan_granted']==0]

datacobination1 = train1
# ----------------------------------- MODELLING-----------------------------------------------------------------------------

from imblearn.over_sampling import SMOTE #reduces oversampling/ technique
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report , f1_score 
from sklearn.tree import DecisionTreeClassifier
from confusionMatrix import plotConfusionMatrix

ywithoutsmote1 = datacobination1['loan_repaid']
Xwithoutsmote1 = datacobination1.iloc[:,0:-1]


# #####  Method 0: without SMOTE---------------------------------------

# In[50]:


X_train, X_val, y_train, y_val = train_test_split(Xwithoutsmote1, ywithoutsmote1, test_size = 0.3, random_state = 0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf.score(X_train, y_train)
clf.score(X_val, y_val)

predictions_ = clf.predict(X_val) 
  
# print classification report 
print('Without imbalance treatment:'.upper())
print(classification_report(y_val, predictions_)) 
print(confusion_matrix(y_val, predictions_))
f1= f1_score(y_val,predictions_, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val == 0)
sum(y_val == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)


# #####  Method1: SMOTE on Train------------------------------------------------------------------

# In[52]:


sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X_train, y_train.ravel()) 

print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_res.shape)) 
print('After OverSampling, y: {}'.format(y_res.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_res == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_res == 0))) 
print('\n')


# In[53]:


#split into 70:30 ratio 
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_res, y_res, 
                    test_size = 0.3, random_state = 0)

#Checking on Validation Set
clf.fit(X_train_res, y_train_res.ravel()) 
clf.score(X_train_res, y_train_res)
clf.score(X_val_res, y_val_res)

predictions = clf.predict(X_val_res) 
  
# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_res, predictions)) 
print(confusion_matrix(y_val_res, predictions))
f1= f1_score(y_val_res,predictions, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val_res,predictions)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val_res == 0)
sum(y_val_res == 1)
sum(predictions == 0)
sum(predictions == 1)

confusion_matrix(y_val_res, predictions)


# #### GRID SEARCH WITH SMOTE---------------------------------------------------

# In[54]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}   X_val, y_val
parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_res,y_train_res)
preds = tree.predict(X_val)
accu = accuracy_score(y_val, preds)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Accuracy:', np.round(accu,3))
print('*'*80)
print('Using best parameters:',tree.best_params_)
print('*'*80)
print(classification_report(y_val, preds))
print('*'*80)
print(confusion_matrix(y_val, preds))
print('*'*80)
y_pred_proba_ = tree.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba_)
auc = roc_auc_score(y_val, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

print('*'*80)


# #### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------

# In[55]:


def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(5, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_train_res,y_train_res, 4)
acc = model.score(X_val, y_val)
print('Using best parameters:',best_param)
print('accuracy:', np.round(acc,3))

## ROC curve
y_pred_proba = model.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba)
auc = metrics.roc_auc_score(y_val, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()


# #### ---------- Random Forest----------------------------------------

# In[56]:


from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [100,300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier()
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    predict_gscv=rf_gscv.predict(X)
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf, rf_gscv.best_params_, rf_gscv, predict_gscv, score_gscv  

print('*'*80)
print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, pred_rf, acc_rf = RF_grid_search(X_train_res,y_train_res, 4)
pred_rf = model_rf.predict(X_val)
acc_rf = model_rf.score(X_val, y_val)
print('Using best parameters:',best_param_rf)
print('accuracy with Gs:', np.round(acc_rf,3))
print('*'*80)
print(classification_report(y_val, pred_rf))
print('*'*80)
print(confusion_matrix(y_val, pred_rf))
print('*'*80)

## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_val)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_val,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_val, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


# ## XG Boost ::

# In[57]:


# import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train_res, y_train_res)

# make predictions for test set
y_pred = model_G.predict(X_val)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_val,y_val)

print('*'*80)
print('SMOTE with XGB:')
print("Accuracy without Gs: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_val, y_pred))
print('*'*80)
print(confusion_matrix(y_val, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_val)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_val, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')


# ####  Feature Importance

# In[77]:


from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)


# In[58]:


# Without SMOTE using XGB
# import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train, y_train)

# make predictions for test set
y_pred = model_G.predict(X_val)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_val,y_val)

print('*'*80)
print('SMOTE with XGB:')
print("Accuracy without Gs: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_val, y_pred))
print('*'*80)
print(confusion_matrix(y_val, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_val)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_val, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')


# ### SVM 

# In[67]:


""" ==================
Now will show SVM data ++++++++++++++++++
"""
df5 = pd.concat([df_nom, dfs_numeric, df_ordinal], axis = 1)
trainsvm=df5[df5['loan_granted']==1]
testsvm=df5[df5['loan_granted']==0]
datasvm = trainsvm


from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve

ywismotesvm = datasvm['loan_repaid']
Xwismotesvm = datasvm.iloc[:,0:-1]
X_train, X_val, y_train, y_val = train_test_split(Xwismotesvm, ywismotesvm, test_size = 0.3, random_state = 0)


# In[70]:


sm = SMOTE(random_state = 2) 
X_sm, y_sm = sm.fit_sample(X_train, y_train) 

print('With imbalance treatment:'.upper())
print('Before SMOTE:',Xwismotesvm.shape, ywismotesvm.shape)
print('After SMOTE, Xs: {}'.format(X_sm.shape)) 
print('After SMOTE, y: {}'.format(y_sm.shape)) 
print("After SMOTE, counts of '1': {}".format(sum(y_sm == 1))) 
print("After SMOTE, counts of '0': {}".format(sum(y_sm == 0))) 
print("Before SMOTE, counts of '1': {}".format(sum(ywismotesvm == 1))) 
print("Before SMOTE, counts of '0': {}".format(sum(ywismotesvm == 0))) 
print('\n')
print('*'*80)


# In[71]:


X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_sm, y_sm, test_size = 0.3, random_state = 0)


# In[72]:


# Naked Modelling 

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train, y_train.ravel()) 
score1 = svm.score(X_train, y_train)
score2 = svm.score(X_val, y_val)
pred = svm.predict(X_val) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy Without SMOTE: ', score2)
print('F1 score:\n', classification_report(y_val, pred)) 
print('*'*80)

# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train, y_train.ravel()) # for probability

pred_proba = svm_p.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  pred_proba)
auc = roc_auc_score(y_val, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr,tpr,label="Without Smote, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()
print('*'*80)


# In[74]:


# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_val_sm, y_val_sm)
score2 = svm.score(X_val, y_val)
pred = svm.predict(X_val) 
  
# print classification report 
print('With SMOTE:'.upper())
print('Validation accuracy: ', score1)
print('test accuracy With SMOTE: ', score2)
print('F1 score:\n', classification_report(y_val, pred)) 
print('*'*80)

# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train_sm, y_train_sm.ravel()) # for probability

pred_proba = svm_p.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  pred_proba)
auc = roc_auc_score(y_val, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr,tpr,label="Smote, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()
print('*'*80)


# ### Logistic Regression

# In[75]:


from sklearn.linear_model import LogisticRegression

lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_train, y_train)
lR.score(X_train, y_train)
acc = lR.score(X_val, y_val)
preds = lR.predict(X_val)
pred_proba = lR.predict_proba(X_val)[::,1]


print('*'*80)
print('Logistic Regression:')
print("Accuracy without LogR: %.2f%%" % (acc * 100.0))
print('*'*80)
#y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  pred_proba)
auc = metrics.roc_auc_score(y_val, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


print(classification_report(y_val, preds)) 
print(confusion_matrix(y_val, preds))
cnf_mat = confusion_matrix(y_val, preds)
f1= f1_score(y_val,preds, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# In[76]:


from sklearn.linear_model import LogisticRegression

lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_train_sm, y_train_sm)
acc = lR.score(X_val, y_val)
preds = lR.predict(X_val)
pred_proba = lR.predict_proba(X_val)[::,1]


print('*'*80)
print('Logistic Regression:')
print("Accuracy without LogR: %.2f%%" % (acc * 100.0))
print('*'*80)
#y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  pred_proba)
auc = metrics.roc_auc_score(y_val, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


print(classification_report(y_val, preds)) 
print(confusion_matrix(y_val, preds))
cnf_mat = confusion_matrix(y_val, preds)
f1= f1_score(y_val,preds, average='micro')
plt.figure()
cnf_mat = confusion_matrix(y_val,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# In[ ]:




