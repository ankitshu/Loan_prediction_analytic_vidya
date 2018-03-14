
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import itertools

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


def plot_confusion_matrix(cm, classes = ['0','1'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[3]:


train=pd.read_csv('train.csv')

train['Gender'].fillna('Male', inplace=True)
train['Married'].fillna('Yes', inplace=True)

train['Self_Employed'].fillna('No', inplace=True)

train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna( train['Loan_Amount_Term'].mean() )

impute = train.pivot_table(values=['LoanAmount'], 
                           index=['Property_Area','Education','Self_Employed','Married'],
                           aggfunc=np.mean)
print("*********",impute)
for i,row in train.loc[train['LoanAmount'].isnull(),:].iterrows():
    ind = tuple([row['Property_Area'],row['Education'],row['Self_Employed']])
    train.loc[i,'LoanAmount'] = impute.loc[ind].values[0]


# In[4]:


print(train['Credit_History'].value_counts())

impute = train.pivot_table(values=['Credit_History'], 
                           index=['Self_Employed','Married','Property_Area','Education','Gender'],
                           aggfunc=np.mean)

#,'ApplicantIncome','CoapplicantIncome','Property_Area'
#,row['ApplicantIncome'],row['CoapplicantIncome'],row['Property_Area']
for i,row in train.loc[train['Credit_History'].isnull(),:].iterrows():
    ind = tuple([row['Self_Employed'],row['Married'],row['Property_Area'],row['Education'],row['Gender']])
    train.loc[i,'Credit_History'] = impute.loc[ind].values[0]

train['Credit_History'].fillna(0, inplace=True)
#train['Credit_History'] = np.where(train['Credit_History'] < 0.85, 0, 1)
print(train['Credit_History'].value_counts())


# In[5]:


train['Gender'] = train['Gender'].map({'Female': 1, 'Male': 0})

train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})

train['Dependents'] = train['Dependents'].str.replace('+', '')
train['Dependents'] = train['Dependents'].map({'0': 0, '1': 1, '2' : 2, '3' : 3})

train['Education'] = train['Education'].map({'Graduate': 1, 'Not Graduate': 0})

train['Self_Employed'] = train['Self_Employed'].map({'No': 0, 'Yes': 1})

train['Property_Area'] = train['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban' : 2})

train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})


# In[6]:


impute = train.pivot_table(values=['Dependents'], 
                           index=['Married', 'Property_Area', 'Gender','Self_Employed'],
                           aggfunc=np.median)

for i,row in train.loc[train['Dependents'].isnull(),:].iterrows():
    ind = tuple([row['Married'],row['Property_Area'],row['Gender'],row['Self_Employed']])
    train.loc[i,'Dependents'] = impute.loc[ind].values[0]


# In[ ]:





# In[7]:


train.apply(lambda x: sum(x.isnull()),axis=0)


# In[8]:


temp = pd.DataFrame()
temp = train
del temp['Loan_ID']
temp['Total_Income'] = temp['ApplicantIncome'] + temp['CoapplicantIncome']
temp['ApplicantIncome'] = np.log(temp['ApplicantIncome'])
temp['LoanAmount'] = np.log(temp['LoanAmount'])
temp['Total_Income'] = np.log(temp['Total_Income'])


# In[9]:


data = temp.drop(['Gender','Education','Self_Employed', 
                  'Loan_Status','Married', 'Dependents','Property_Area',
                  'LoanAmount',
                  'Loan_Amount_Term', 'ApplicantIncome','CoapplicantIncome'], axis=1)
print(data.columns)
target = pd.DataFrame() 
target = temp['Loan_Status']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=0)
kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)


# In[11]:


y_test.value_counts()


# In[13]:


modelDT = DecisionTreeClassifier(criterion='gini',
                                 max_depth=1,
                                 min_samples_split=40,
                                 max_features=2)
modelDT.fit(X_train,y_train)
score_val = np.mean(cross_val_score(modelDT, X_train, y_train, cv=kf, scoring='accuracy') )
predicted = modelDT.predict(X_test)

print('cross_val_score {}\n'.format(score_val))
print('classification_report\n', metrics.classification_report(y_test, predicted))
print('accuracy_score {}\n'.format(metrics.accuracy_score(y_test, predicted)))

cm = metrics.confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm)

pd.crosstab(y_test,predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[13]:


modelRF = RandomForestClassifier(n_estimators=50, 
                                 min_samples_leaf=10,
                                 min_samples_split=30, 
                                 max_depth=1, 
                                 max_features=2)

#modelRF = RandomForestClassifier(n_estimators=30, 
#                                 min_samples_split=30, 
#                                 max_depth=5, 
#                                 max_features=4)
modelRF.fit(X_train,y_train)

score_val = np.mean(cross_val_score(modelRF, X_train, y_train, cv=kf, scoring='accuracy') )
predicted = modelRF.predict(X_test)

print('cross_val_score {}\n'.format(score_val))
print('classification_report\n', metrics.classification_report(y_test, predicted))
print('accuracy_score {}\n'.format(metrics.accuracy_score(y_test, predicted)))

cm = metrics.confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm)


# In[14]:


modelLR = LogisticRegression(warm_start=True,
                             max_iter=200,
                             fit_intercept=True)
modelLR.fit(X_train,y_train)

score_val = np.mean(cross_val_score(modelLR, X_train, y_train, cv=kf, scoring='accuracy') )
predicted = modelLR.predict(X_test)

print('cross_val_score {}\n'.format(score_val))
print('classification_report\n', metrics.classification_report(y_test, predicted))
print('accuracy_score {}\n'.format(metrics.accuracy_score(y_test, predicted)))


cm = metrics.confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm)


# In[15]:


modelGNB = GaussianNB()
modelGNB.fit(X_train,y_train)

score_val = np.mean(cross_val_score(modelGNB, X_train, y_train, cv=kf, scoring='accuracy') )
predicted = modelGNB.predict(X_test)

print('cross_val_score {}\n'.format(score_val))
print('classification_report\n', metrics.classification_report(y_test, predicted))
print('accuracy_score {}\n'.format(metrics.accuracy_score(y_test, predicted)))


cm = metrics.confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm)


# In[105]:


modelABC = AdaBoostClassifier(base_estimator=modelRF, n_estimators=50, learning_rate=.03, algorithm="SAMME.R")
modelABC.fit(X_train,y_train)

score_val = np.mean(cross_val_score(modelABC, X_train, y_train, cv=kf, scoring='accuracy') )
predicted = modelABC.predict(X_test)

print('cross_val_score {}\n'.format(score_val))
print('classification_report\n', metrics.classification_report(y_test, predicted))
print('accuracy_score {}\n'.format(metrics.accuracy_score(y_test, predicted)))


cm = metrics.confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm)


# In[ ]:





# In[ ]:





# In[ ]:




# SUBMISSION





# In[119]:


test=pd.read_csv('E:/myAnacondaProj/test.csv')
print("Table Missing Values")
test.apply(lambda x: sum(x.isnull()),axis=0)


# In[120]:


test['Gender'].fillna('Male', inplace=True)
test['Married'].fillna('Yes', inplace=True)
test['Self_Employed'].fillna('No', inplace=True)


# In[121]:


impute = test.pivot_table(values=['LoanAmount'], 
                           index=['Property_Area','Education','Self_Employed','Married'],
                           aggfunc=np.mean)

for i,row in test.loc[test['LoanAmount'].isnull(),:].iterrows():
    ind = tuple([row['Property_Area'],row['Education'],row['Self_Employed'],row['Married']])
    test.loc[i,'LoanAmount'] = impute.loc[ind].values[0]
    
test['Loan_Amount_Term'].fillna(np.mean(test['Loan_Amount_Term']), inplace=True)


# In[122]:


impute = test.pivot_table(values=['Credit_History'], 
                           index=['Self_Employed','Married','Property_Area','Education','Gender'],
                           aggfunc=np.mean)

for i,row in test.loc[test['Credit_History'].isnull(),:].iterrows():
    ind = tuple([row['Self_Employed'],row['Married'],row['Property_Area'],row['Education'],row['Gender']])
    test.loc[i,'Credit_History'] = impute.loc[ind].values[0]

#test['Credit_History'] = np.where(test['Credit_History'] < 0.85, 0, 1)



# In[123]:


test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})

test['Married'] = test['Married'].map({'Yes': 1, 'No': 0})

test['Dependents'] = test['Dependents'].str.replace('+', '')
test['Dependents'] = test['Dependents'].map({'0': 0, '1': 1, '2' : 2, '3' : 3})

test['Education'] = test['Education'].map({'Graduate': 1, 'Not Graduate': 0})

test['Self_Employed'] = test['Self_Employed'].map({'No': 0, 'Yes': 1})

test['Property_Area'] = test['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban' : 2})


# In[124]:


impute = test.pivot_table(values=['Dependents'], 
                           index=['Married'],
                           aggfunc=np.median)

#,'ApplicantIncome','CoapplicantIncome','Property_Area'
#,row['ApplicantIncome'],row['CoapplicantIncome'],row['Property_Area']

for i,row in test.loc[test['Dependents'].isnull(),:].iterrows():
    ind = tuple([row['Married']])
    test.loc[i,'Dependents'] = impute.loc[ind].values[0]
    
#test['Dependents'].fillna(0, inplace=True)


# In[125]:


test.apply(lambda x: sum(x.isnull()),axis=0)


# In[126]:


sub_pred = test.copy()


# In[127]:


sub_pred['Total_Income'] = sub_pred['ApplicantIncome'] + sub_pred['CoapplicantIncome']


# In[128]:


sub_pred['LoanAmount'] = np.log(sub_pred['LoanAmount'])


# In[129]:


sub_pred['Total_Income'] = np.log(sub_pred['Total_Income'])


# In[130]:


sub_pred_data = sub_pred.drop(['Gender','Education','Self_Employed', 
                                'Loan_ID','Married', 'Dependents','Property_Area',
                                'CoapplicantIncome', 'LoanAmount',
                                'Loan_Amount_Term', 'ApplicantIncome'],axis=1)
print(len(sub_pred_data.columns))
sub_pred_data.apply(lambda x: sum(x.isnull()),axis=0)


# In[131]:


predictedRF_S = modelRF.predict(sub_pred_data)
predictedLR_S = modelLR.predict(sub_pred_data)
predictedABC_S = modelABC.predict(sub_pred_data)
predictedGNB_S = modelGNB.predict(sub_pred_data)
predictedDT_S = modelDT.predict(sub_pred_data)


# In[132]:


compare = sub_pred.drop(['CoapplicantIncome','Gender', 'Married','Dependents','Property_Area',
                         'Education','Loan_Amount_Term','Self_Employed','ApplicantIncome',
                         'LoanAmount','Credit_History','Total_Income'],axis=1)

compare['Loan_Status'] = predictedRF_S + predictedLR_S + predictedABC_S  + predictedGNB_S + predictedDT_S
compare.head()


# In[133]:


compare['Loan_Status'].value_counts()


# In[134]:


compare['Loan_Status'] = compare['Loan_Status'].map({5:'Y', 4:'Y', 3:'Y', 2: 'Y', 1:'N', 0:'N'})


# In[135]:


compare.to_csv('my_submission4.csv', index=False)


# In[136]:


compare.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:





# In[ ]:




