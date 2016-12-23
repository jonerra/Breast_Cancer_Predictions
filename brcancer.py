import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df = df.drop(['id'], 1)
# df.replace('?', -99999, inplace=True)
df.replace('?', np.nan, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].fillna(df['bare_nuclei'].mode()[0])

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
# clf = svm.LinearSVC()
# clf = RandomForestClassifier(n_estimators=100)
# clf = GradientBoostingClassifier(n_estimators=100)
# clf = AdaBoostClassifier(n_estimators=100)
# clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
