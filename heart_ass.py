# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:37:59 2022

@author: ACER
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle



import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%%
CSV_PATH = os.path.join(os.getcwd(),'heart.csv')

#%% EDA

# Step 1 Data Loading
df = pd.read_csv(CSV_PATH)
df_backup = df.copy()

# Step 2 Data Inspection
df.head(10)
df.tail(10)

df.info() #to check if theres Nan (no missing values)
df.describe().T ## to check percentile,mean, min-max, count
df.duplicated().sum() # check for total duplicated data
df[df.duplicated()] # check for total duplicated values

df.columns #Get column names
df.boxplot() # to check summary of a set of data

# to visualize your data
#to see number each sex in data set
# categorical data
#for categorical data
#to see the number in the dataset
categorical=['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
for cat in categorical:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

#for continuous data
#to see the distribution in the dataset
continuous=['age','trtbps','chol','thalachh','oldpeak']
for con in continuous:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

# Step 3 Data Cleaning
# Got no Nan

#drop duplicated data 
df = df.drop_duplicates()
df.info() # all duplicated has been removed



# Step 4 Features Selection
#continuous vs categorical data using LogisticRegression

con_column=['age','trtbps','chol','thalachh','oldpeak']

# # score is accuracy
# logreg= LogisticRegression()
# logreg.fit(np.expand_dims(df['chol'],axis=1),df['sex'])
# logreg.score(np.expand_dims(df['chol'],axis=1),df['sex']) #0.7 #score is for accuracy

for con in con_column:
    logreg= LogisticRegression()
    logreg.fit(np.expand_dims(df[con],axis=1),df['output'])
    print(con + ' '+ str(logreg.score(np.expand_dims(df[con],axis=1),df['output'])))


X= df.loc[:,['age','trtbps','chol','thalachh','oldpeak','sex','cp','fbs',
              'restecg','exng','slp','caa','thall','output']] 
y= df['output'] 

#cramers corrected stat

for cat in categorical:
    print(cat)
    confussion_mat = pd.crosstab(df[cat], df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))
    


# # Step 5 Preprocessing


X = df.loc[:,['thalachh','oldpeak','thall']]
y = df.loc[:,'output']


X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size=0.3,
                                                  random_state=123)

#%% pipeline



#Logistic Regression
step_mms_lr =Pipeline([('mmsscaler', MinMaxScaler()),
            ('lr', LogisticRegression())])

step_ss_lr = Pipeline([('sscaler', StandardScaler()),
            ('lr', LogisticRegression())])

# Random Forest
step_mms_rf = Pipeline([('mmsscaler', MinMaxScaler()),
            ('rf', RandomForestClassifier())])

step_ss_rf = Pipeline([('sscaler', StandardScaler()),
            ('rf', RandomForestClassifier())])

# Decision Tree
step_mms_tree = Pipeline([('mmsscaler', MinMaxScaler()),
            ('tree', DecisionTreeClassifier())])

step_ss_tree = Pipeline([('sscaler', StandardScaler()),
            ('tree', DecisionTreeClassifier())])

#knn 
step_mms_knn = Pipeline([('mmsscaler', MinMaxScaler()),
            ('knn', KNeighborsClassifier())])

step_ss_knn = Pipeline([('sscaler', StandardScaler()),
            ('knn', KNeighborsClassifier())])


#%% creating pipelines
# pipelines
pipelines = [step_mms_lr,step_ss_lr,step_mms_rf,step_ss_rf,step_mms_tree,
             step_ss_tree,step_mms_knn,step_ss_knn]

for pipe in pipelines:
    print(pipe)
    pipe.fit(X_train,y_train)

# model evaluation

best_accuracy = 0
for i,model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
  
print('The best pipeline for this heart dataset for predicting heart attack is {} with accuracy of {}'.
      format(best_pipeline, best_accuracy))


#%% This is to fine tune the model
# Steps for rf

step_rf = [('mmsscaler Scaler', MinMaxScaler()),
           ('RandomForestClassifier', RandomForestClassifier(random_state=123))] 

pipeline_rf = Pipeline(step_rf)

# number of trees
grid_param = [{'RandomForestClassifier':[RandomForestClassifier()],
                'RandomForestClassifier__n_estimators':[10,100,1000],
                'RandomForestClassifier__max_depth':[None,5,15]}]



gridsearch = GridSearchCV(pipeline_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)
best_model.score(X_test,y_test)
print(best_model.score(X_test, y_test))
print(best_model.best_index_)
print(best_model.best_params_)



#%% Model analysis

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_true = y_test
y_pred = best_model.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))


#%% MODEL SAVING


PKL_PATH = os.path.join(os.getcwd(),'best_pipeline.pkl')


with open(PKL_PATH,'wb') as file:
    pickle.dump(best_model,file) #to be load in deploy

































