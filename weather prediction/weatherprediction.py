import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('weatherAUS.csv')

data.dropna(inplace=True)

data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d", errors = "coerce")
data["Date"]=pd.to_numeric(data["Date"], downcast="float")

data["RainToday"] = pd.get_dummies(data["RainToday"], drop_first = True)
data["RainTomorrow"] = pd.get_dummies(data["RainTomorrow"], drop_first = True)

le = preprocessing.LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])
data['WindDir9am'] = le.fit_transform(data['WindDir9am'])
data['WindDir3pm'] = le.fit_transform(data['WindDir3pm'])
data['WindGustDir'] = le.fit_transform(data['WindGustDir'])

X= data.iloc[:, :22]
Y= data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
GradientBoostingClassifierScore = gb.score(X_test,Y_test)
print("Accuracy obtained by Gradient Boosting Classifier model:",GradientBoostingClassifierScore*100)


pickle.dump(gb,open('pred.pkl','wb'))