# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:46:12 2019

@author: ateeb ahmad
"""
#importing important library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading the data
data=pd.read_excel("Data_Dictionary.xlsx")
dataset=pd.read_excel("Loan.xlsx")

#NOW we will drop the column with half missing values as many columns has large no of missing values
half_count=len(dataset)/2
dataset=dataset.dropna(thresh=half_count,axis=1)


#after dropping the values we drop the column  url and desc because it is also useless
dataset=dataset.drop(['url','desc'],axis=1)

#preview the data
print(data.shape[0])
print(data.columns.tolist())
data.head()

#now making a new dataframe in ordeer to store the the name datatype description of the colunms by joining the two dataframes
loans_dtypes = pd.DataFrame(dataset.dtypes,columns=['dtypes']) #made the new dataframe and made the first entry of datatypes 
loans_dtypes = loans_dtypes.reset_index()#adiing a index to the dataframe
loans_dtypes['name'] = loans_dtypes['index'] #making a copy of the column conatianbg the names as the name
loans_dtypes = loans_dtypes[['name','dtypes']]# now keeping only the name and datatypes column


#now joing the two dataframes
loans_dtypes['first value'] = dataset.loc[0].values
data=data.rename(columns={"LoanStatNew":"name","Description":"description"})
preview = loans_dtypes.merge(data, on='name',how='left')

#now deving the dataframe into three equal parts of19 each and removing the featues that are unnnecessary or which will not be there ata the time of application of loan
preview[:19]
drop_list = ['id','member_id','funded_amnt','funded_amnt_inv',
'int_rate','sub_grade','emp_title','issue_d']
dataset=dataset.drop(drop_list,axis=1)

#dropping unnecessary  columns
drop_cols = [ 'zip_code','out_prncp','out_prncp_inv',
'total_pymnt','total_pymnt_inv']
dataset=dataset.drop(drop_cols, axis=1)


drop_cols = ['total_rec_prncp','total_rec_int',
'total_rec_late_fee','recoveries',
'collection_recovery_fee']
dataset=dataset.drop(drop_cols, axis=1)


########################################################



#now fisrt consider what type of values are there in the loan_status and after that only considering the fully paid or charged off and assigining the string the numerical values
dataset["loan_status"].value_counts()
dataset = dataset[(dataset["loan_status"] == "Fully Paid") |(dataset["loan_status"] == "Charged Off")]
dataset= dataset.replace({"loan_status":{"Fully Paid":1,"Charged Off":0}})


dataset=dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]


for col in dataset.columns:
    if(len(dataset[col].unique())<4):
        print(dataset[col].value_counts())

dataset["term"].value_counts()
dataset= dataset.replace({"term":{" 36 months":1," 60 months":2}})

null_counts = dataset.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))


dataset = dataset.drop("pub_rec_bankruptcies",axis=1)

dataset= dataset.dropna(axis =0)

object_columns_df = dataset.select_dtypes(include=['object'])
print(object_columns_df.loc[0])


dataset =dataset.drop(['title'], axis=1)

from sklearn.preprocessing import LabelEncoder 
Labelencoder_X=LabelEncoder()

dataset.iloc[:,3]=Labelencoder_X.fit_transform(dataset.iloc[:,3])
dataset.iloc[:,4]=Labelencoder_X.fit_transform(dataset.iloc[:,4])
dataset.iloc[:,5]=Labelencoder_X.fit_transform(dataset.iloc[:,5])
dataset.iloc[:,7]=Labelencoder_X.fit_transform(dataset.iloc[:,7])
dataset.iloc[:,9]=Labelencoder_X.fit_transform(dataset.iloc[:,9])


#now dropping unnecessary columns 

drop_cols = ['last_credit_pull_d','addr_state','earliest_cr_line']
dataset=dataset.drop(drop_cols,axis=1)
dataset=dataset.drop(["last_pymnt_d"],axis=1)



#now our data is clean and ready for processing and now dividing the data into X,Y ,Y cotaining the final loan status
Y=dataset.iloc[:,8].values 
X=dataset.drop(["loan_status"],axis=1)

#splitting the data

from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)


#now applying various algo for data prediction

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 10)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X,Y)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X,Y)



from sklearn.ensemble import RandomForestClassifier
ran_clf = RandomForestClassifier(n_jobs = -1)
ran_clf.fit(X_train, y_train)
ran_clf.score(X_train, y_train)
ran_clf.score(X_test, y_test)
ran_clf.score(X, Y)




from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_train, y_train)
gnb.score(X_test, y_test)
gnb.score(X, Y)








