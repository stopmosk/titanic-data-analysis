#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_train.head(3)


# In[3]:


y_train = X_train.Survived


# In[4]:


X_train.isna().mean()  # Доля пропусков в данных (NaN значений) по столбцам


# In[5]:


X_test.isna().mean()  # Доля пропусков в данных (NaN значений) по столбцам


# In[6]:


med = (X_train.Age.median() + X_test.Age.median()) // 2
X_train.fillna({'Age': med}, inplace=True)
X_test.fillna({'Age': med}, inplace=True)
X_train.isna().mean()  # Проверили


# In[7]:


X_train.index = X_train['PassengerId']
X_test.index = X_test['PassengerId']
X_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_train.head()


# In[8]:


X_train = pd.get_dummies(X_train)  # One-hot encoding
X_test = pd.get_dummies(X_test)  # One-hot encoding
X_train.head()


# In[9]:


X_train.drop(['Sex_male', 'Embarked_S'], axis=1, inplace=True)
X_test.drop(['Sex_male', 'Embarked_S'], axis=1, inplace=True)
X_train.head()


# In[10]:


X_test.isna().mean()  # Проверили
med = (X_train.Fare.median() + X_test.Fare.median()) // 2
X_train.fillna({'Fare': med}, inplace=True)
X_test.fillna({'Fare': med}, inplace=True)
X_test.isna().mean()  # Проверили


# In[11]:


clf = DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'], 
              'max_depth': range(1, 30), 
              'min_samples_split': range(2, 10), 
              'min_samples_leaf': range(1,10)}
grid_clf = GridSearchCV(clf, parameters, cv=5, scoring='f1')


# In[12]:


grid_clf.fit(X_train, y_train);
grid_clf.best_params_  # Вот и они, наши лучшие параметры


# In[13]:


best_clf = grid_clf.best_estimator_
best_clf.fit(X_train, y_train)


# In[14]:


best_clf.score(X_train, y_train)


# In[26]:


y_pred = X_test.drop(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Embarked_C', 'Embarked_Q'], axis=1)
y_pred['Survived'] = best_clf.predict(X_test)
y_pred.head()


# In[25]:


y_pred.to_csv('gender_submission.csv')


# In[17]:


y_predicted_prob = best_clf.predict_proba(X_train)   # Вытаскиваем вероятности отнесения пассажира к тому или иному классу (а не просто Dead/Survived)
pd.Series(y_predicted_prob[:, 1]).hist();

