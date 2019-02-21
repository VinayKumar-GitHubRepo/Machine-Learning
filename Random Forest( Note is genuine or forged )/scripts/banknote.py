import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"banknote.txt",sep=",",
                 names = ['var','skewness','kurtosis','entropy','class'])

print(df.head())
fig = plt.figure()
plt.scatter(df['var'],df['skewness'],c=df['class'],alpha=0.5)
plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.show()
X = df.drop(['class'],axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
clf_gini = RandomForestClassifier(criterion = "gini")
clf_gini.fit(X_train, y_train)

clf_entropy = RandomForestClassifier(criterion = "entropy")
clf_entropy.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)
print("Accuracy is ",accuracy_score(y_test, y_pred_gini)*100)
print("Accuracy is ",accuracy_score(y_test, y_pred_entropy)*100)
