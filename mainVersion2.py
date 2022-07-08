

import pandas as pd

data_set1 = pd.read_csv("csv/matches.csv")

print(data_set1.info())
teamNames = data_set1.team1.unique()

print(teamNames)

"""Replace with shorthand"""

shortHand = ['SRH', 'MI', 'GL', 'RPS', 'RCB',
                           'KKR', 'DD', 'KXIP', 'CSK', 'RR', 'DC',
                           'KTK', 'PW', 'RPS']

data_set1.replace(teamNames,shortHand,inplace=True)

print(data_set1.head(5))



"""Replace NAN in winner with draw"""

data_set1['winner'].fillna('Draw', inplace=True)


"""Fill dubai in venue if valueis NAN"""




data_set1['city'].fillna('Dubai',inplace=True)
#print(data_set1[pd.isnull(data_set1['city'])]) # 7 rows is empty , NAN

"""Describe for full dataset1"""

# print(data_set1.describe())

#print(shortHand.index('CSK'))


data_set = (data_set1[['team1','team2','city','toss_decision','toss_winner','venue','winner']])
data_set2=pd.DataFrame(data_set)

print(data_set2.describe())

most_winner = data_set2["winner"].value_counts(sort=True)

most_tossWinner  = data_set2["toss_winner"].value_counts(sort=True)

print(most_winner,most_tossWinner)

data_set2["winner"].hist(bins=40,grid=True,legend=True)

import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(121)
#
# ax1.set_xlabel("Teams")
# ax1.set_ylabel("No of times")
# ax1.set_title("True winners of most season")
# most_winner.plot(kind='bar')
# plt.legend()
#
# #plt.show()
#
# import seaborn as sb
#
# ax = sb.displot(most_winner,legend=True)
# plt.show()


"""checking finally is any field has null value"""

print(data_set2.isnull().sum()) #  dataset2.apply(lambda x: sum(x.isnull()),axis=0)

"""Data preprocessing is complete . lets starts with model prediction"""

#label encoder
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

print(data_set2.dtypes)
for i in data_set2.columns:

    data_set2[i]=label.fit_transform(data_set2[i])

print(data_set2.columns)
print(data_set2.describe())

from datetime import datetime

# dataset_1["date"] = dataset_1["date"].apply(
#     lambda x: datetime.strptime(x, '%Y-%m-%d'))  # according to input we need to add DDMMYY wont work

"""Splitting train and test test """

#as of now not needed

"""Build ML models"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import numpy as np

def predictModel(models,data_set,x,y):

    models.fit(data_set[x],data_set[y])
    predict_y = models.predict(data_set[x])
    accuracy = metrics.accuracy_score(predict_y,data_set[y])
    print("Accuracy :",accuracy)
    kf = KFold(data_set.shape[0],n_splits=5)
    error =[]
    for train , test in kf:
        train_X = data_set[x].iloc[train,:]
        train_y = data_set[y].iloc[train]
        models.fit(train_X,train_y)
        error.append(models.score(data_set[x].iloc[test,:],data_set[y].iloc[test]))
    print("Cross Validation Score :",np.mean(error))


X=['team1','team2','toss_winner']

y=["winner"]

model_regressor = RandomForestRegressor()

#model_regressor.fit(data_set2[['team1','team2','toss_winner']],data_set2['winner'])
predictModel(models=model_regressor,data_set=data_set2,x=['team1','team2','toss_winner'],y=['winner'])

