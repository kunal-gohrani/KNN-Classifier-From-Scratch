import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNeighborsClassifier import KNeighborsClassifier

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

# splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# Scaling the X_train and X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# knn.predict(np.array([[1,2],[3,4]]))

def user_input():
    k = int(input('Enter K value: '))
    knn = KNeighborsClassifier(k=k, X_train=X_train, y_train=y_train)
    knn.fit()
    while True:
        age = int(input('Enter Age: '))
        salary = float(input('Enter Salary: '))
        data = np.array([[age,salary]])
        data = scaler.transform((data))
        answer = knn.predict(data)[0]
        if answer==0:
            print('Will not Purchase')
        else:
            print('Will Purchase')
        quit = input('Do You want to quit? Enter Y/N: ')
        if quit==str.lower('Y'):
            print('Thanks for using the program')
            exit(0)
user_input()
