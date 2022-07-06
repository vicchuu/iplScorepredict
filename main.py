import pandas as pd
import numpy as np


columns_to_read = ["date","venue","team1","team2","toss_winner","toss_decision","winner","win_by_runs","season",
                   "win_by_wickets"]
dataset_1 = pd.read_csv("csv/matches.csv",usecols=columns_to_read)

#.rename(columns = {'test':'TEST'}, inplace = True)
#arr = [dataset_1.iloc[0,:]]


present_teams = ["Sunrisers Hyderabad","Mumbai Indians","Royal Challengers Bangalore",
                 "Kolkata Knight Riders","Delhi Daredevils","Kings XI Punjab",
                 "Chennai Super Kings","Rajasthan Royals"]


"""Dataset-1 ends here..."""
dataset_1 = dataset_1[(dataset_1["team1"].isin(present_teams)) & (dataset_1["team2"].isin(present_teams))]
# print(len(dataset_1.team1.unique()),len(dataset_1.team2.unique()))
# print(dataset_1.shape)
#print

battingTeam=[]
bowlingTeam =[]
for key,value in dataset_1.iterrows():
    #print(value)
    if (value.toss_winner and (value.toss_decision == "bat")):

        # print("T1:",(value.team1),"   T2 :",value.team2, "  tW :",value.toss_winner," ")
        # print(f"Id : {key}  Batting team...{value.toss_decision}")

        battingTeam.append(value.toss_winner)
        bowling =""
        if value.team1 == value.toss_winner:
            bowling = value.team2
        else:
            bowling = value.team1
        bowlingTeam.append(bowling)
    else:
        bowlingTeam.append(value.toss_winner)
        batting =""

        if value.team1 == value.toss_winner:
            batting =value.team2
        else:
            batting = value.team1
        battingTeam.append(batting)



# print(dataset_1.shape)
# print(len(battingTeam))
# print(len(bowlingTeam))
dataset_1["batting_team"] = battingTeam
dataset_1["bowling_team"] = bowlingTeam

#print(dataset_1.head(5))
dataset_1= dataset_1.drop(["team1","team2"],axis=1)


#toCSV = dataset_1.to_csv("newDataset.csv")

"""Checking any null values in all column"""

#print(dataset_1.isnull().sum()) # except 2 winner option everything took place

"""grouping by its column and their extraction"""

#info abt data set
"""Data columns (total 9 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   date            451 non-null    object
 1   toss_winner     451 non-null    object
 2   toss_decision   451 non-null    object
 3   winner          449 non-null    object
 4   win_by_runs     451 non-null    int64 
 5   win_by_wickets  451 non-null    int64 
 6   venue           451 non-null    object
 7   batting_team    451 non-null    object
 8   bowling_team    451 non-null    object
dtypes: int64(2), object(7)"""
#print(dataset_1.info())


"""Describe()
       win_by_runs  win_by_wickets   bcos only 2 column is in INTEGER (int64)
count   451.000000      451.000000
mean     14.394678        3.368071
std      24.671520        3.434110
min       0.000000        0.000000
25%       0.000000        0.000000
50%       0.000000        4.000000
75%      21.500000        6.500000
max     146.000000       10.000000
"""

#print(dataset_1.describe())

"""view the unique values of each column"""

dataset_1 = dataset_1.drop_duplicates(keep=False)

#print(dataset_1.shape) # same shape so no duplicates


# for col in dataset_1:
#     print(dataset_1[col].unique())


"""Grouping the data set """

grp = dataset_1.groupby('season')['winner'].value_counts()

"""for need of time series and to split data into year basd we use datetime """
from datetime import datetime

dataset_1["date"] =dataset_1["date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d')) #according to input we need to add DDMMYY wont work

"""Splitting trainning data and test data"""


"""Label encoding for binary values --> """
from sklearn.preprocessing import LabelEncoder

import sklearn.preprocessing as pss
label_Encod= ["toss_decision","winner","toss_winner","batting_team","bowling_team","venue"] # 0 ,1 will be converted

encoding = pss.LabelEncoder()
for a in label_Encod:
    dataset_1[a] = encoding.fit_transform(dataset_1[a])


#print(dataset_1.head(5))
dataset_2 = pd.get_dummies(dataset_1)  #One hot encoding

#print(dataset_2.iloc[1,:])


# print(dataset_2.head(5))
#
xtrain= dataset_2[dataset_2['date'].dt.year<=2016]
#
xtest = dataset_2[dataset_2['date'].dt.year>=2017]
# #
# #
# #

ytrain = xtrain["winner"]
#ytrain= ytrain[:,:]

#ytrain.to_csv("1.csv")

print(ytrain.shape)

xtrain = xtrain.drop(["date","winner"],axis=1)
#print("$$$$ ",xtrain.columns)
#
ytest = xtest["winner"]
#ytest = ytest[0]

#a = { n:v for n,v in zip(ytest,range(0,31)) }

#print(ytest[0])

xtest = xtest.drop(["date","winner"],axis=1)
#




from sklearn.preprocessing import StandardScaler

sf_scaler = StandardScaler()

#xtrain = pd.DataFrame(sf_scaler.fit_transform(xtrain))
#xtest = pd.DataFrame(sf_scaler.fit_transform(xtest))
#sf_scaler.fit(xtest)


"""Model-1  leads woth Linear Regression """

from sklearn.linear_model import LinearRegression

import math
model1 = LinearRegression()

model1.fit(xtrain,ytrain)

from sklearn.metrics import r2_score ,confusion_matrix

predict_y = (model1.predict(xtest))

predict_y = [round(x)  for x in predict_y]

#print("R2 score is :",(r2_score(ytest,predict_y)))

# print(np.asarray(predict_y).shape)
# print(ytest.shape)
# print(ytrain[0:10])
ss = (confusion_matrix(ytest,predict_y))

#print(ss)  #predictio is so poor so , better try wth some other prediction

"""lets try our SVM"""

from sklearn.tree import DecisionTreeRegressor

model2 = DecisionTreeRegressor()
mm = model2.fit(xtrain,ytrain)

pred_model2 = model2.predict(xtest)

score = r2_score(ytest,pred_model2)

#print("SVM r2 score :",score)

model_confusion = confusion_matrix(ytest,pred_model2)

#print("model2 confusion matrix :",model_confusion)

#print(predict_y.summary())
#print(mm.summary())


pred = xtrain.iloc[113]
ans_pred = ytrain.iloc[113]

print(pred)
print(ans_pred)
print(xtest.shape , pred)
act_pred = model2.predict([[pred]])


