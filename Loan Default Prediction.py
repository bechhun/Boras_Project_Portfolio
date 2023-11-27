#!/usr/bin/env python
# coding: utf-8

# In[255]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[256]:


import csv
open(r'C:\Users\Bora\OneDrive - Push Thru Analytics\Desktop\loan default project.csv')


# In[257]:


df = pd.read_csv(r'C:\Users\Bora\OneDrive - Push Thru Analytics\Desktop\loan default project.csv', encoding = 'latin-1')


# In[258]:


df.head()


# In[259]:


#Looking at data types
df.info()


# In[260]:


#looking for NA values
df.isnull().sum()


# In[261]:


df['loanAmount_log'] = np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins =20)


# In[262]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[ ]:





# In[263]:


#Replacing all NA values

df['Gender'].fillna(df['Gender'].mode()[0], inplace =True)
df['Married'].fillna(df['Married'].mode()[0], inplace =True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace =True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace =True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace =True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace =True)

df.isnull().sum()


# In[264]:


df.columns[1:5]


# In[265]:


df.columns[9:11]


# In[266]:


df.columns[13:15]


# In[267]:


x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values

x


# In[268]:


y


# In[269]:


print("per of missinf gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[270]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df, palette = 'Set1')


# In[271]:


print("number of people who take loan as group by marital status:")
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df, palette = 'Set1')


# In[272]:


print("number of people who take loan as group by Dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df, palette = 'Set2')


# In[273]:


print("number of people who take loan as group by Self Employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df, palette = 'Set3')


# In[274]:


print("number of people who take loan as group by Loan Amount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df, palette = 'Set2')


# In[275]:


print("number of people who take loan as group by Credit History:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df, palette = 'Set1')


# In[276]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x = LabelEncoder()


# In[277]:


for i in range(0,5):
    X_train[:,i]= Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]= Labelencoder_x.fit_transform(X_train[:,7])
    
X_train


# In[278]:


Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)

y_train


# In[279]:


for i in range(0,5):
    X_test[:,i] = Labelencoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = Labelencoder_x.fit_transform(X_test[:,7])
    
X_test


# In[280]:


Labelencoder_y = LabelEncoder()

y_test = Labelencoder_y.fit_transform(Y_test)

y_test


# In[281]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# In[282]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)


# In[283]:


from sklearn import metrics
y_pred = rf_clf.predict(X_test)

print("acc of random forest classifier is", metrics.accuracy_score(y_pred, y_test))

y_pred


# In[284]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)


# In[285]:


y_pred = nb_clf.predict(X_test)
print('acc of naive bayes is ', metrics.accuracy_score(y_pred, y_test))


# In[286]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)


# In[287]:


y_pred = dt_clf.predict(X_test)
print("Acc of Decision Tree is", metrics.accuracy_score(y_pred, y_test))


# In[288]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train)


# In[289]:


y_pred = kn_clf.predict(X_test)
print("Acc of KNN is", metrics.accuracy_score(y_pred, y_test))


# In[290]:


y_pred


# In[291]:


y_test


# In[292]:


df


# In[311]:


new_applicant = np.array([[1,1,5,1,4,0,1,10]])
Test_predict = dt_clf.predict(new_applicant)
Test_predict

