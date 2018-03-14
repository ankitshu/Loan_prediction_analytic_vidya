'''
Created on 06-Sep-2017

@author: ankit
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
#sklearn modelling
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


train=pd.read_csv("train.csv")
train[train.dtypes[(train.dtypes=="float64")|(train.dtypes=="int64")].index.values].hist(figsize=[11,11])
plt.show()

print("*********quantative*********")
print(train.describe())
print("*********qualitative*********")
print(train.describe(include=[object]))
#data cleaning
print('total no of misiing values')
print(train.apply(lambda x: sum(x.isnull()),axis=0))
#filling the data
train['Gender'].fillna('Male', inplace=True)
train['Married'].fillna('Yes', inplace=True)

train['Self_Employed'].fillna('No', inplace=True)

train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna( train['Loan_Amount_Term'].mean() )

impute=train.pivot_table(values=["LoanAmount"],
                         index=['Property_Area','Education','Self_Employed','Married'],
                         aggfunc=np.mean)
print(impute)
for i,row in train.loc[train["LoanAmount"].isnull(),:].iterrows():
    ind = tuple([row['Property_Area'],row['Education'],row['Self_Employed'],row["Married"]])
    train.loc[i,"LoanAmount"]=impute.loc[ind].values[0]

train["Credit_History"].fillna(1,inplace=True)
print(train['Credit_History'].value_counts())
print(train.apply(lambda x: sum(x.isnull()),axis=0))

train["Totalincome"]=train["LoanAmount"]+train["ApplicantIncome"]
train["LoanAmount"]=np.log(train["LoanAmount"])
train["ApplicantIncome"]=np.log(train["ApplicantIncome"])    
train["Totalincome"]=np.log(train["Totalincome"])
#convert categorial value into NumericAlign
train['Gender'] = train['Gender'].map({'Female': 1, 'Male': 0})

train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})

train['Dependents'] = train['Dependents'].str.replace('+', '')
train['Dependents'] = train['Dependents'].map({'0': 0, '1': 1, '2' : 2, '3' : 3})

train['Education'] = train['Education'].map({'Graduate': 1, 'Not Graduate': 0})

train['Self_Employed'] = train['Self_Employed'].map({'No': 0, 'Yes': 1})

train['Property_Area'] = train['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban' : 2})

train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})
print(train.apply(lambda x:sum(x.isnull()),axis=0))
# filling dependents

impute=train.pivot_table(values=["Dependents"],
                         index=['Married', 'Property_Area', 'Gender','Self_Employed'],
                         aggfunc=np.median)
print(impute)
for i,row in train.loc[train["Dependents"].isnull(),:].iterrows():
    ind=tuple([row["Married"],row["Property_Area"],row["Gender"],row["Self_Employed"]])
    train.loc["Dependents"]=impute.loc[ind].values[0]


train[train.dtypes[(train.dtypes=="float64")|(train.dtypes=="int64")].index.values].hist(figsize=[11,11])
plt.show()
'''
correlatkon table
corr=train[['Loan_Status',"Property_Area","Self_Employed","Dependents","Gender","Married"]].corr()
sns.heatmap(corr,annot=True, fmt=".2f",cmap="YlGnBu")
plt.show()
'''
print("**********logistic regression model*************")
#logistic regression model
#Generic function for making a classification model and accessing performance:
def classification_model(model, train, predictors, outcome):
    #Fit the model:
    model.fit(train[predictors],train[outcome])
  
  #Make predictions on training set:
    predictions = model.predict(train[predictors])
  
  #Print accuracy
    accuracy = metrics.accuracy_score(predictions,train[outcome])
    
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, train,predictor_var,outcome_var)


print("*******************************Random Forest**************")

model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model,train,predictor_var,outcome_var)

print("***************decision tree**********")
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model,train,predictor_var,outcome_var)
print("**************implement**************")

#implementation
test=pd.read_csv("test.csv")
print("*********quantative*********")
print(test.describe())
print("*********qualitative*********")
print(test.describe(include=[object]))
#data cleaning
print('total no of misiing values')
print(test.apply(lambda x: sum(x.isnull()),axis=0))
#filling the data
test['Gender'].fillna('Male', inplace=True)
test['Married'].fillna('Yes', inplace=True)

test['Self_Employed'].fillna('No', inplace=True)

test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna( test['Loan_Amount_Term'].mean() )
impute=test.pivot_table(values=["LoanAmount"],
                         index=['Property_Area','Education','Self_Employed','Married'],
                         aggfunc=np.mean)
for i,row in test.loc[test["LoanAmount"].isnull(),:].iterrows():
    ind = tuple([row['Property_Area'],row['Education'],row['Self_Employed'],row["Married"]])
    test.loc[i,"LoanAmount"]=impute.loc[ind].values[0]
    
test["Credit_History"].fillna(1,inplace=True)

test["Totalincome"]=test["LoanAmount"]+test["ApplicantIncome"]
test["LoanAmount"]=np.log(test["LoanAmount"])
test["Totalincome"]=np.log(test["Totalincome"])
# test["ApplicantIncome"]=np.log(test["ApplicantIncome"])    

#convert categorial value into NumericAlign

test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})

test['Married'] = test['Married'].map({'Yes': 1, 'No': 0})

test['Dependents'] = test['Dependents'].str.replace('+', '')
test['Dependents'] = test['Dependents'].map({'0': 0, '1': 1, '2' : 2, '3' : 3})

test['Education'] = test['Education'].map({'Graduate': 1, 'Not Graduate': 0})

test['Self_Employed'] = test['Self_Employed'].map({'No': 0, 'Yes': 1})

test['Property_Area'] = test['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban' : 2})

# filling dependents

impute=test.pivot_table(values=["Dependents"],
                         index=['Married', 'Property_Area', 'Gender','Self_Employed'],
                         aggfunc=np.median)
for i,row in test.loc[test["Dependents"].isnull(),:].iterrows():
    ind=tuple([row["Married"],row["Property_Area"],row["Gender"],row["Self_Employed"]])
    test.loc[i,"Dependents"]=impute.loc[ind].values[0]

print("***************check nan values count************")
print(test.apply(lambda x: sum(x.isnull()),axis=0))

#PredictionResults
print("**************lets do it***********")
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
predictions = model.predict(test[predictor_var])
print(predictions)
test['Loan_Status'] = predictions
test['Loan_Status'] = test['Loan_Status'].map({0:'N',1:'Y'})


test.to_csv('testprediction.csv', index=False)

